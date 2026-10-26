# SPDX-License-Identifier: Apache-2.0

"""Free-energy, common-tangent and differentiation checks with independent constraints."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import root
from scipy.special import expit, logit

from atmodeller.sci_utils import GAS_CONSTANT as R
from atmodeller.thermodata.mixing import SubregularSolution


def test_ideal_and_pure_limits():
    model = SubregularSolution.regular_solution(0)
    x = jnp.array([0.0, 0.1, 0.5, 0.9, 1.0])
    g = model.gibbs_energy(2000.0, 1.0, x)
    assert g[jnp.array([0, 4])].tolist() == [0.0, 0.0]
    np.testing.assert_allclose(g[2], -R * 2000 * np.log(2), rtol=1e-14)
    np.testing.assert_array_equal(model.log_activity_coefficients(2000.0, 1.0, x), 0)
    np.testing.assert_allclose(
        jnp.exp(model.log_activities(2000.0, 1.0, x)), jnp.stack((1 - x, x), axis=-1)
    )
    assert not model.coexistence(2000.0, 1.0).has_gap
    assert model.coexistence(2000.0, 1.0).valid


def test_chemical_potentials_and_gibbs_duhem():
    model = SubregularSolution(-2000.0, 110000.0, temperature_scale=5000, pressure_scale=-4e5)
    t, p = 2500.0, 25000.0

    def total_gibbs(n):
        return jnp.sum(n) * model.gibbs_energy(t, p, n[1] / jnp.sum(n))

    for x in (0.05, 0.3, 0.7, 0.98):
        n = jnp.array([1 - x, x])
        partial_molar = jax.grad(total_gibbs)(n)
        log_activity = model.log_activities(t, p, x, equilibrate=False)
        np.testing.assert_allclose(partial_molar, R * t * log_activity, rtol=1e-12, atol=1e-10)
        dmu = jax.jacfwd(lambda z: model.log_activities(t, p, z, equilibrate=False))(x)
        assert float(jnp.dot(n, dmu)) == pytest.approx(0, abs=1e-12)
        second = jax.grad(jax.grad(lambda z: model.gibbs_energy(t, p, z)))(x)
        assert float(model.curvature(t, p, x)) == pytest.approx(float(second), rel=1e-12)


def test_regular_solution_spinodal_and_critical_temperature():
    t = 2000.0
    omega = 3 * R * t
    model = SubregularSolution.regular_solution(omega)
    d = model.diagnostics(t, 1.0, 0.3)
    endpoints = (1 + np.array([-1, 1]) * np.sqrt(1 - 2 * R * t / omega)) / 2
    np.testing.assert_allclose([d["spinodal_lower"], d["spinodal_upper"]], endpoints, rtol=1e-13)
    assert bool(d["spinodally_unstable"])
    assert not bool(d["metastable"])
    assert bool(model.diagnostics(t, 1.0, 0.1)["metastable"])
    assert not bool(model.diagnostics(t, 1.0, 0.01)["equilibrium_two_phase"])
    for hot in (omega / (2 * R), 2 * omega / R):
        c = model.coexistence(hot, 1.0)
        assert not c.has_gap
        assert c.valid
        assert c.upper_fraction(0.3) == 0


@pytest.mark.parametrize("t,p", [(3000.0, 10000.0), (3975.0, 10000.0), (3500.0, 50000.0)])
def test_common_tangent_is_global_support_and_conserves_components(t, p):
    model = SubregularSolution.hydrogen_silicate_rogers25()
    c = eqx.filter_jit(model.coexistence)(t, p)
    assert c.has_gap and c.valid
    assert float(c.residual) < 1e-12
    x = jnp.linspace(0, 1, 2001)
    g = model.gibbs_energy(t, p, x)
    tangent = R * t * ((1 - x) * c.log_activity[0] + x * c.log_activity[1])
    assert float(jnp.min(g - tangent)) >= -1e-8
    bulk = (c.lower + c.upper) / 2
    fraction = c.upper_fraction(bulk)
    assert float(fraction) == pytest.approx(0.5)
    assert float((1 - fraction) * c.lower + fraction * c.upper) == pytest.approx(float(bulk))
    np.testing.assert_allclose(
        model.log_activities(t, p, jnp.array([c.lower, bulk, c.upper])),
        jnp.broadcast_to(c.log_activity, (3, 2)),
        atol=1e-12,
    )
    # Independently solve the paper's dimensionless free energy in composition
    # coordinates with SciPy, rather than reusing any implementation helper.
    a = -6260 * (1 - t / 4670 + p / -350000) / (R * t)
    b = 786000 * (1 - t / 4670 + p / -350000) / (R * t)

    def potentials(z):
        f = (1 - z) * np.log1p(-z) + z * np.log(z) + z * (1 - z) * (a * (1 - z) + b * z)
        df = np.log(z / (1 - z)) + a + 2 * (b - 2 * a) * z + 3 * (a - b) * z * z
        return np.array([f - z * df, f + (1 - z) * df])

    answer = root(
        lambda pair: potentials(expit(pair[0])) - potentials(expit(pair[1])),
        logit(np.array([float(c.lower), float(c.upper)])) + np.array([0.01, -0.01]),
        tol=1e-10,
    )
    assert answer.success
    np.testing.assert_allclose([c.lower, c.upper], expit(answer.x), rtol=1e-10)


def test_rogers_pressure_conversion_and_component_order():
    model = SubregularSolution.hydrogen_silicate_rogers25()
    t, p, x = 3975.0, 10000.0, 0.6
    direct = R * t * ((1 - x) * np.log(1 - x) + x * np.log(x))
    direct += x * (1 - x) * (-6260 * (1 - x) + 786000 * x) * (1 - t / 4670 + 1 / -35)
    assert float(model.gibbs_energy(t, p, x)) == pytest.approx(direct, rel=1e-13)
    c = model.coexistence(t, p)
    np.testing.assert_allclose(
        [c.lower, c.upper], [0.43491265779265, 0.94691945544375], atol=1e-12
    )
    reversed_model = SubregularSolution(
        786000, -6260, temperature_scale=4670, pressure_scale=-350000
    )
    reverse = reversed_model.coexistence(t, p)
    np.testing.assert_allclose(
        [reverse.lower, reverse.upper], [1 - c.upper, 1 - c.lower], atol=1e-12
    )
    np.testing.assert_allclose(reverse.log_activity, c.log_activity[::-1], atol=1e-12)


def test_jit_vmap_and_parameter_broadcasting():
    model = SubregularSolution.hydrogen_silicate_rogers25()
    temperatures = jnp.array([3000.0, 3975.0, 6000.0])
    batch = eqx.filter_jit(model.coexistence)(temperatures, 10000.0)
    mapped = jax.jit(jax.vmap(lambda t: model.coexistence(t, 10000.0)))(temperatures)
    np.testing.assert_allclose(batch.logits, mapped.logits, atol=1e-12)
    np.testing.assert_array_equal(batch.has_gap, [True, True, False])
    np.testing.assert_array_equal(batch.valid, True)
    array_model = SubregularSolution(0.0, jnp.array([0.0, 2e5]))
    assert array_model.coexistence(2000.0, 1.0).logits.shape == (2, 2)


@pytest.mark.parametrize("index", [0, 1])
def test_implicit_binodal_and_activity_derivatives(index):
    model = SubregularSolution.hydrogen_silicate_rogers25()
    tp = jnp.array([3975.0, 10000.0])

    def values(v):
        c = model.coexistence(v[0], v[1])
        return jnp.concatenate(
            (c.logits, model.log_activities(v[0], v[1], 0.65), model.spinodal(v[0], v[1]))
        )

    derivative = jax.jit(jax.jacfwd(values))(tp)[:, index]
    step = jnp.eye(2)[index] * 0.01
    difference = (values(tp + step) - values(tp - step)) / 0.02
    np.testing.assert_allclose(derivative, difference, rtol=2e-7, atol=1e-10)
    reverse = jax.jacrev(values)(tp)[:, index]
    np.testing.assert_allclose(reverse, derivative, rtol=1e-12, atol=1e-14)


def test_near_critical_and_extremely_dilute_binodals():
    for interaction in (2.000002, 3.0, 100.0):
        model = SubregularSolution.regular_solution(interaction * R * 2000)
        c = model.coexistence(2000.0, 1.0)
        assert c.has_gap and c.valid
        assert float(c.lower) < 0.5 < float(c.upper)
        assert np.isfinite(np.asarray(c.log_activity)).all()
        assert float(c.residual) < 1e-10
    # Deliberately exceed the documented numerical log-odds range. A failed
    # tangent must never be substituted by a homogeneous equilibrium result.
    invalid = SubregularSolution.regular_solution(800 * R * 2000)
    assert not invalid.coexistence(2000.0, 1.0).valid
    assert np.isnan(invalid.log_activities(2000.0, 1.0, 0.5)).all()


@pytest.mark.parametrize("t,p", [(0.0, 1.0), (-1.0, 1.0), (2000.0, -1.0), (np.nan, 1.0)])
def test_invalid_thermodynamic_inputs(t, p):
    model = SubregularSolution.regular_solution(50000)
    assert not model.coexistence(t, p).valid
    assert np.isnan(model.log_activities(t, p, 0.5)).all()


def test_invalid_composition_and_no_gap_derivative():
    model = SubregularSolution.regular_solution(0)
    for x in (-0.1, 1.1, np.nan):
        assert np.isnan(model.gibbs_energy(2000.0, 1.0, x))
        assert np.isnan(model.curvature(2000.0, 1.0, x))
        assert np.isnan(model.log_activities(2000.0, 1.0, x)).all()
        assert np.isnan(model.coexistence(2000.0, 1.0).upper_fraction(x))
    np.testing.assert_array_equal(
        jax.jacfwd(lambda t: model.coexistence(t, 1.0).logits)(2000.0), 0
    )


def test_parameters_remain_differentiable_equinox_leaves():
    model = SubregularSolution(40000.0, 80000.0)
    derivative = eqx.filter_grad(lambda m: m.log_activities(2000.0, 1.0, 0.5).sum())(model)
    for field in ("interaction_0", "interaction_1"):
        plus = eqx.tree_at(lambda m: getattr(m, field), model, getattr(model, field) + 0.01)
        minus = eqx.tree_at(lambda m: getattr(m, field), model, getattr(model, field) - 0.01)
        difference = (
            plus.log_activities(2000.0, 1.0, 0.5).sum()
            - minus.log_activities(2000.0, 1.0, 0.5).sum()
        ) / 0.02
        np.testing.assert_allclose(getattr(derivative, field), difference, rtol=1e-6)
    # In a fully miscible state no inactive NaN sentinel contaminates AD.
    hot_derivative = jax.jacfwd(lambda t: model.log_activities(t, 1.0, 0.4))(10000.0)
    assert np.isfinite(hot_derivative).all()
