# SPDX-License-Identifier: Apache-2.0

"""Binary mixing thermodynamics, spinodals, and equilibrium common tangents.

All energies are per mole of *components*, temperatures are K, pressures are bar,
and ``x`` is the mole fraction of component 1 in the ordered pair (0, 1).
These are local equilibrium calculations, not a model of decomposition kinetics.
"""

from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy
from jaxtyping import Array, ArrayLike, Bool

from atmodeller.jax_utils import FloatArray, as_j64
from atmodeller.sci_utils import GAS_CONSTANT


def _bisect(
    function: Callable[[FloatArray], FloatArray], lower: FloatArray, upper: FloatArray
) -> FloatArray:
    """Bisect an increasing, bracketed scalar function, with broadcasting."""

    def step(_, bounds):
        lo, hi = bounds
        mid = lo + (hi - lo) / 2
        negative = function(mid) < 0
        return jnp.where(negative, mid, lo), jnp.where(negative, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 64, step, (lower, upper))
    return lo + (hi - lo) / 2


def _curvature(a: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    """Second composition derivative of g_mix / RT."""
    return 1 / (x * (1 - x)) + 2 * b - 4 * a + 6 * (a - b) * x


def _third_derivative(a: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    return -1 / x**2 + 1 / (1 - x) ** 2 + 6 * (a - b)


def _spinodal_limits_impl(a: FloatArray, b: FloatArray) -> FloatArray:
    # The fourth derivative is positive, so the curvature has exactly one minimum.
    # Bounded composition roots here are checked below; the binodal uses log odds
    # to retain accuracy for exponentially dilute coexisting components.
    lower = jnp.full_like(a, 1e-14)
    upper = jnp.full_like(a, 1 - 1e-14)
    minimum = _bisect(lambda x: _third_derivative(a, b, x), lower, upper)
    has_gap = _curvature(a, b, minimum) < 0
    left = _bisect(lambda x: -_curvature(a, b, x), lower, minimum)
    right = _bisect(lambda x: _curvature(a, b, x), minimum, upper)
    return jnp.where(has_gap[..., None], jnp.stack((left, right), axis=-1), 0.5)


_spinodal_limits = jax.custom_jvp(_spinodal_limits_impl)


@_spinodal_limits.defjvp
def _spinodal_limits_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    x = _spinodal_limits(a, b)
    gap = x[..., 0] < x[..., 1]
    denominator = jnp.where(gap[..., None], _third_derivative(a[..., None], b[..., None], x), 1.0)
    dx = -((-4 + 6 * x) * da[..., None] + (2 - 6 * x) * db[..., None]) / denominator
    return x, jnp.where(gap[..., None], dx, 0.0)


def _spinodal(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray, Bool[Array, "..."]]:
    limits = _spinodal_limits(a, b)
    left, right = limits[..., 0], limits[..., 1]
    return left, right, left < right


def _excess_potentials(a: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    """Partial molar excess Gibbs energies divided by RT, in component order."""
    mu0 = x**2 * (2 * a - b + 2 * (b - a) * x)
    mu1 = (1 - x) ** 2 * (a + 2 * (b - a) * x)
    return jnp.stack((mu0, mu1), axis=-1)


def _potentials_from_logit(a: FloatArray, b: FloatArray, q: FloatArray) -> FloatArray:
    x, y = jax.nn.sigmoid(q), jax.nn.sigmoid(-q)
    # Evaluate both complementary fractions and their logs independently. Near a
    # pure endpoint, 1 - sigmoid(q) can round to zero even when exp(-q) is resolved.
    mu0 = -jax.nn.softplus(q) + x**2 * (2 * a - b + 2 * (b - a) * x)
    mu1 = -jax.nn.softplus(-q) + y**2 * (a + 2 * (b - a) * x)
    return jnp.stack((mu0, mu1), axis=-1)


def _binodal_logits_impl(a: FloatArray, b: FloatArray) -> FloatArray:
    """Common tangent on the two stable branches; no equal-composition root."""
    sl, sr, has_gap = _spinodal(a, b)
    # Use a well-conditioned dummy problem where the true mixture has no gap.
    a_work, b_work = jnp.where(has_gap, a, 3.0), jnp.where(has_gap, b, 3.0)
    sl = jnp.where(has_gap, sl, (1 - jnp.sqrt(1 / 3)) / 2)
    sr = jnp.where(has_gap, sr, (1 + jnp.sqrt(1 / 3)) / 2)
    ql, qr = jnp.log(sl) - jnp.log1p(-sl), jnp.log(sr) - jnp.log1p(-sr)
    qmin, qmax = jnp.full_like(a, -700.0), jnp.full_like(a, 700.0)

    def slope(q):
        x = jax.nn.sigmoid(q)
        return q + a_work + 2 * (b_work - 2 * a_work) * x + 3 * (a_work - b_work) * x**2

    def endpoints(m):
        left = _bisect(lambda q: slope(q) - m, qmin, ql)
        right = _bisect(lambda q: slope(q) - m, qr, qmax)
        return left, right

    def tangent_difference(m):
        left, right = endpoints(m)
        # mu0 = g/RT - x*m is the tangent intercept. Its right-minus-left
        # difference decreases strictly with m, hence this sign for bisection.
        return (
            _potentials_from_logit(a_work, b_work, left)[..., 0]
            - _potentials_from_logit(a_work, b_work, right)[..., 0]
        )

    m = _bisect(tangent_difference, slope(qr), slope(ql))
    left, right = endpoints(m)
    return jnp.where(has_gap[..., None], jnp.stack((left, right), axis=-1), 0.0)


# Construct the JAX callable after definition so the test suite's import-time
# type instrumentation wraps the function, rather than the custom_jvp object.
_binodal_logits = jax.custom_jvp(_binodal_logits_impl)


@_binodal_logits.defjvp
def _binodal_logits_jvp(primals, tangents):
    """Implicit derivative of equality of both component chemical potentials.

    Differentiating the comparisons in bisection would give incorrect derivatives.
    The common-tangent Jacobian supplies derivatives with respect to both model
    parameters, and thus also temperature and pressure. At a critical point the
    two-phase derivative is singular; on the no-gap branch the sentinel is constant.
    """
    a, b = primals
    da, db = tangents
    q = _binodal_logits(a, b)
    x, y = jax.nn.sigmoid(q), jax.nn.sigmoid(-q)
    gap = q[..., 0] < q[..., 1]
    c = 1 + x * y * (2 * b[..., None] - 4 * a[..., None] + 6 * (a - b)[..., None] * x)
    j00, j01 = -x[..., 0] * c[..., 0], x[..., 1] * c[..., 1]
    j10, j11 = y[..., 0] * c[..., 0], -y[..., 1] * c[..., 1]
    dmu0 = x**2 * (2 * y * da[..., None] + (2 * x - 1) * db[..., None])
    dmu1 = y**2 * ((1 - 2 * x) * da[..., None] + 2 * x * db[..., None])
    d0, d1 = dmu0[..., 0] - dmu0[..., 1], dmu1[..., 0] - dmu1[..., 1]
    determinant = jnp.where(gap, j00 * j11 - j01 * j10, 1.0)
    dl = (-d0 * j11 + j01 * d1) / determinant
    dr = (-j00 * d1 + j10 * d0) / determinant
    dq = jnp.where(gap[..., None], jnp.stack((dl, dr), axis=-1), 0.0)
    return q, dq


class BinaryCoexistence(eqx.Module):
    """Equilibrium coexistence at specified temperature and pressure.

    ``logits`` stores the two compositions as log(x/(1-x)), preserving dilute
    limits even when a mole fraction rounds to unity. Where there is no gap the
    logits are zero, both compositions are 0.5, and ``has_gap`` is false. These
    are sentinels, not coexistence endpoints. Always inspect ``has_gap`` and
    ``valid``. Invalid numerical results must not be used as an equilibrium.
    """

    logits: FloatArray
    has_gap: Bool[Array, "..."]
    valid: Bool[Array, "..."]
    log_activity: FloatArray
    """Common component activities, natural log, shape (..., 2); NaN without a gap"""
    residual: FloatArray
    """Maximum chemical potential mismatch divided by RT (zero without a gap)"""

    @property
    def lower(self) -> FloatArray:
        """Component-1 mole fraction of the component-0-rich phase."""
        return jax.nn.sigmoid(self.logits[..., 0])

    @property
    def upper(self) -> FloatArray:
        """Component-1 mole fraction of the component-1-rich phase."""
        return jax.nn.sigmoid(self.logits[..., 1])

    def contains(self, x: ArrayLike) -> Bool[Array, "..."]:
        """Whether the bulk composition lies strictly inside the coexistence gap."""
        x = as_j64(x)
        return self.has_gap & self.valid & (x > self.lower) & (x < self.upper)

    def upper_fraction(self, x: ArrayLike) -> FloatArray:
        """Mole fraction of the upper-composition phase from the lever rule.

        Returns zero outside the gap, where a second phase is not present, and
        NaN for an invalid calculation. This is a *phase amount*, not a component
        abundance or a mass fraction.
        """
        x = as_j64(x)
        width = jnp.where(self.has_gap, self.upper - self.lower, 1.0)
        fraction = jnp.where(self.contains(x), (x - self.lower) / width, 0.0)
        return jnp.where(self.valid & (x >= 0) & (x <= 1), fraction, jnp.nan)


class SubregularSolution(eqx.Module):
    r"""A thermodynamically consistent binary subregular solution.

    .. math::
        g_{\rm mix} = RT[(1-x)\ln(1-x)+x\ln x]
          + x(1-x)[A(1-x)+Bx](1-T/C+P/D).

    ``x`` is component 1's mole fraction. Pure-component standard Gibbs energies
    are supplied separately by Atmodeller's species and must use compatible
    component and standard-state conventions. Linear standard-state terms do
    not change a common tangent's compositions or the spinodal.

    Args:
        interaction_0: A (J/mol of components).
        interaction_1: B (J/mol of components).
        temperature_scale: C (K); infinity disables temperature scaling.
        pressure_scale: D (bar); infinity disables pressure scaling. May be negative.

    Inputs broadcast elementwise. Invalid T <= 0, P < 0, nonfinite T/P or x
    outside [0,1] produce NaN. No extrapolation validity is implied by a finite
    answer. A binary parameter set is not transferable to a different mixture.
    """

    interaction_0: FloatArray
    interaction_1: FloatArray
    temperature_scale: FloatArray
    pressure_scale: FloatArray

    def __init__(
        self,
        interaction_0: ArrayLike,
        interaction_1: ArrayLike,
        *,
        temperature_scale: ArrayLike = jnp.inf,
        pressure_scale: ArrayLike = jnp.inf,
    ):
        self.interaction_0 = as_j64(interaction_0)
        self.interaction_1 = as_j64(interaction_1)
        self.temperature_scale = as_j64(temperature_scale)
        self.pressure_scale = as_j64(pressure_scale)

    @classmethod
    def regular_solution(cls, interaction: ArrayLike) -> Self:
        """Symmetric regular solution with g_excess = interaction*x*(1-x), J/mol."""
        return cls(interaction, interaction)

    @classmethod
    def hydrogen_silicate_rogers25(cls) -> Self:
        """H2-MgSiO3 parameterization of Rogers et al. (2025), equation 7.

        Component 0 is MgSiO3 and component 1 is H2 (formula-unit mole fractions).
        This reproduces the published phenomenological mixing model, not a new
        calibration to the final 2026 DFT paper. The published D=-35 GPa is
        converted to -350000 bar. See the spinodal guide for provenance and scope.
        """
        return cls(-6.26e3, 7.86e5, temperature_scale=4.67e3, pressure_scale=-3.5e5)

    def _parameters(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        t, p = as_j64(temperature), as_j64(pressure)
        factor = (1 - t / self.temperature_scale + p / self.pressure_scale) / (GAS_CONSTANT * t)
        valid = (t > 0) & (p >= 0) & jnp.isfinite(t) & jnp.isfinite(p)
        a, b = jnp.broadcast_arrays(self.interaction_0 * factor, self.interaction_1 * factor)
        return jnp.where(valid, a, jnp.nan), jnp.where(valid, b, jnp.nan)

    def gibbs_energy(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Homogeneous molar Gibbs energy of mixing (J/mol); finite at x=0 and 1."""
        a, b = self._parameters(temperature, pressure)
        x = as_j64(x)
        f = xlogy(x, x) + xlogy(1 - x, 1 - x) + x * (1 - x) * (a * (1 - x) + b * x)
        return jnp.where((x >= 0) & (x <= 1), GAS_CONSTANT * temperature * f, jnp.nan)

    def curvature(self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike) -> FloatArray:
        """Homogeneous d²g_mix/dx² (J/mol); negative means spinodal instability."""
        a, b = self._parameters(temperature, pressure)
        x = as_j64(x)
        value = GAS_CONSTANT * temperature * _curvature(a, b, x)
        return jnp.where((x >= 0) & (x <= 1), value, jnp.nan)

    def spinodal(self, temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
        """Lower and upper spinodal compositions, shape (..., 2), or NaN if absent.

        Implicit derivatives are defined away from a critical point. These are
        local-stability limits, not the coexisting compositions (binodal).
        """
        a, b = self._parameters(temperature, pressure)
        lower, upper, gap = _spinodal(a, b)
        return jnp.where(gap[..., None], jnp.stack((lower, upper), axis=-1), jnp.nan)

    def log_activity_coefficients(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Homogeneous ln(gamma_0, gamma_1), derived from the same Gibbs energy."""
        a, b = self._parameters(temperature, pressure)
        x = as_j64(x)
        coefficients = _excess_potentials(a, b, x)
        return jnp.where(((x >= 0) & (x <= 1))[..., None], coefficients, jnp.nan)

    def coexistence(self, temperature: ArrayLike, pressure: ArrayLike) -> BinaryCoexistence:
        """Compute the binodal by equality of both component chemical potentials.

        Uses bracketed solves on the two stable branches, avoiding the trivial
        equal-composition root. The calculation is JIT/vmap compatible and has
        implicit derivatives with respect to its parameters, T and P. A mismatch
        exceeding 1e-8 in mu/RT, a lost bracket, or the log-odds bound +/-700
        marks the result invalid. It must then not be used as a phase equilibrium.
        """
        a, b = self._parameters(temperature, pressure)
        q = _binodal_logits(a, b)
        sl, sr, gap = _spinodal(a, b)
        left = _potentials_from_logit(a, b, q[..., 0])
        right = _potentials_from_logit(a, b, q[..., 1])
        residual = jnp.where(gap, jnp.max(jnp.abs(left - right), axis=-1), 0.0)
        brackets = (jax.nn.sigmoid(q[..., 0]) < sl) & (jax.nn.sigmoid(q[..., 1]) > sr)
        resolved = (residual < 1e-8) & brackets & jnp.all(jnp.abs(q) < 699.0, axis=-1)
        edges_resolved = (
            (_curvature(a, b, as_j64(1e-14)) > 0)
            & (_curvature(a, b, as_j64(1 - 1e-14)) > 0)
            & (_third_derivative(a, b, as_j64(1e-14)) < 0)
            & (_third_derivative(a, b, as_j64(1 - 1e-14)) > 0)
        )
        valid = jnp.isfinite(a) & jnp.isfinite(b) & edges_resolved & (~gap | resolved)
        activity = jnp.where((gap & valid)[..., None], (left + right) / 2, jnp.nan)
        return BinaryCoexistence(q, gap, valid, activity, residual)

    def log_activities(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        x: ArrayLike,
        *,
        equilibrate: bool = True,
    ) -> FloatArray:
        """Log component activities, with equilibrium demixing enabled by default.

        Set ``equilibrate=False`` only to examine a constrained homogeneous state,
        which can be metastable or unstable. Common-tangent activities are constant
        across the coexistence gap. No extra solubility law is applied.
        """
        x = as_j64(x)
        ideal = jnp.stack((jnp.log1p(-x), jnp.log(x)), axis=-1)
        homogeneous = ideal + self.log_activity_coefficients(temperature, pressure, x)
        if not equilibrate:
            return homogeneous
        coexistence = self.coexistence(temperature, pressure)
        values = jnp.where(
            coexistence.contains(x)[..., None], coexistence.log_activity, homogeneous
        )
        return jnp.where(coexistence.valid[..., None], values, jnp.nan)

    def diagnostics(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> dict[str, Array]:
        """Binary phase diagnostics; all entries are scalar or elementwise batches.

        Stability refers to the *homogeneous* free energy, even when equilibrium
        activities use its convex envelope. The lever-rule fraction describes
        the equilibrium end state, not a decomposition rate or elapsed progress.
        Spinodal endpoints are NaN when absent; no-gap binodal sentinels are hidden.
        """
        x = as_j64(x)
        a, b = self._parameters(temperature, pressure)
        sl, sr, gap = _spinodal(a, b)
        coexistence = self.coexistence(temperature, pressure)
        curvature = self.curvature(temperature, pressure, x)
        separated = coexistence.contains(x)
        return {
            "bulk_component_1_mole_fraction": x,
            "homogeneous_curvature_J_per_mol": curvature,
            "spinodally_unstable": curvature < 0,
            "metastable": separated & (curvature >= 0),
            "equilibrium_two_phase": separated,
            "valid": coexistence.valid & (x >= 0) & (x <= 1),
            "spinodal_lower": jnp.where(gap, sl, jnp.nan),
            "spinodal_upper": jnp.where(gap, sr, jnp.nan),
            "binodal_lower": jnp.where(gap, coexistence.lower, jnp.nan),
            "binodal_upper": jnp.where(gap, coexistence.upper, jnp.nan),
            "equilibrium_upper_phase_mole_fraction": coexistence.upper_fraction(x),
            "common_tangent_residual": coexistence.residual,
        }
