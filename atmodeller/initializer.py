# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Initializers for the solver's initial guess

This module provides the concrete implementations of
:class:`~atmodeller.initializer_base.InitializerProtocol`:

- :class:`HeuristicInitializer`: Uses the heuristic guess from
  :func:`generate_auto_initial_guess`
- :class:`ElementPotentialInitializer`: Solves an ideal-gas equilibrium in the element
  potentials. This is the default initializer of :class:`~atmodeller.classes.EquilibriumModel`.

It also contains the heuristic used by :class:`HeuristicInitializer`. All heuristic functions are
designed to be compatible with both :func:`jax.vmap`, as used by the engine and solver routines,
and with explicit batched input arrays, as used by output routines. This means that each function
should correctly handle both single-instance and batched input, broadcasting and returning outputs
with shapes consistent with the input batch dimensions. This ensures seamless integration with
both vectorized and batch-processing workflows throughout the codebase.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float, Integer

from atmodeller.engine import compute_implied_log_stability
from atmodeller.jax_utils import FloatArray
from atmodeller.parameters import Parameters

LOG_TRACE_VALUE: float = -20.0
"""Small trace value (in log space) to assign to species that have a negligible element budget"""
CONDENSATE_ALLOCATION_FRACTION: float = 0.5
"""Fraction of the element budget to allocate to predicted-stable condensates

This value, although arbitrary to some extent, prevents condensates from being allocated an 
excessively large portion of the element budget, which can drive gas species to unrealistically low
initial abundances.
"""


def max_moles_by_limiting_element(
    formula_matrix: Float[Array, "n_elements n_species"],
    element_abundance: Float[Array, "... n_elements"],
    mask: Bool[Array, "... n_species"],
) -> Float[Array, "... n_species"]:
    """Maximum moles by limiting element for species in ``mask``

    For each species in ``mask``, the mole count is estimated by asking: given the available
    element budget, how many moles of this species could be formed if that element were shared
    equally among all masked species that contain it? The tightest such constraint across all
    elements that appear in the species formula determines the estimate. Returns ``NaN`` for
    species not in ``mask`` or not constrained by any available element.

    Args:
        formula_matrix: Matrix of elemental formulas for each species
        element_abundance: Element abundance. ``NaN`` for unconstrained elements.
        mask: Boolean mask selecting the species to allocate budget to.

    Returns:
        Per-species mole estimates; ``NaN`` where unconstrained.
    """
    constrained_element: Bool[Array, "... n_elements"] = ~jnp.isnan(element_abundance)

    # Broadcast all masks to (..., n_elements, n_species)
    fm_mask: Bool[Array, "1 n_elements n_species"] = (formula_matrix > 0)[None, ...]
    mask_b: Bool[Array, "... 1 n_species"] = mask[..., None, :]
    constrained_element_b: Bool[Array, "... n_elements 1"] = constrained_element[..., :, None]

    stoich_total: Float[Array, "... n_elements"] = jnp.sum(
        formula_matrix[None, ...] * mask_b, axis=-1
    )
    safe_stoich: Float[Array, "... n_elements"] = jnp.where(stoich_total > 0, stoich_total, 1.0)
    share: Float[Array, "... n_elements"] = element_abundance / safe_stoich

    is_constrained: Bool[Array, "... n_elements n_species"] = (
        fm_mask & mask_b & constrained_element_b
    )
    # jax.debug.print("is_constrained = {out}", out=is_constrained)
    implied: Float[Array, "... n_elements n_species"] = jnp.where(
        is_constrained, share[..., :, None], jnp.nan
    )
    # jax.debug.print("implied = {out}", out=implied)

    max_moles_by_limiting_element: Float[Array, "... n_species"] = jnp.nanmin(implied, axis=-2)
    # jax.debug.print("max_moles_by_limiting_element = {out}", out=max_moles_by_limiting_element)

    # An extra dimension was introduced to ensure correct broadcasting for both 1-D and 2-D
    # (batched) cases, but we can now squeeze it back out for the single-case scenario to be
    # consistent with the shape of the elemental abundance input.
    return jnp.squeeze(max_moles_by_limiting_element)


def allocate_element_budget(
    parameters: Parameters,
    element_abundance: Float[Array, "... n_elements"],
    condensate_stable_mask: Bool[Array, "... n_species"],
) -> tuple[Float[Array, "... n_species"], Float[Array, "... n_species"]]:
    """Allocates the element budget between predicted-stable condensates and other species.

    This function first assigns as much of each element as possible to the species predicted to be
    stable condensates, according to the limiting-reagent principle. It then computes the remaining
    element budget and allocates it to the non-condensate species, again using the limiting-reagent
    logic. Finally, fallback logic is applied to assign small nonzero values or the geometric mean.

    Args:
        parameters: Parameters
        element_abundance: Available abundance of each element
        condensate_stable_mask: Boolean mask indicating which species are predicted-stable
            condensates

    Returns:
        Tuple of two arrays:
            - n_condensate: Moles allocated to each condensate species
            - n_other: Moles allocated to each non-condensate species
    """
    formula_matrix: Float[Array, "n_elements n_species"] = jnp.asarray(
        parameters.reaction_system.formula_matrix, dtype=float
    )
    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )

    # Allocate element budget to predicted-stable condensates
    n_condensate: Float[Array, "... n_species"] = max_moles_by_limiting_element(
        formula_matrix, element_abundance, condensate_stable_mask
    )
    # jax.debug.print("n_condensate = {out}", out=n_condensate)

    n_condensate = n_condensate * CONDENSATE_ALLOCATION_FRACTION

    # Assign fallback values to unconstrained (NaN) entries after stable-condensate allocation.
    # Species with NaN allocation are considered unstable or absent for element accounting and are
    # assigned a small trace value (exp(LOG_TRACE_VALUE)). This ensures they do not consume element
    # budget and prevents NaN propagation. Any values assigned to non-condensate species here are
    # ignored in subsequent steps by masking.
    n_condensate_fallback: Float[Array, "... n_species"] = jnp.where(
        jnp.isnan(n_condensate), jnp.exp(LOG_TRACE_VALUE), n_condensate
    )
    # jax.debug.print("n_condensate_fallback = {out}", out=n_condensate_fallback)

    # Total moles of each element consumed by the allocated condensate budget. Note here that the
    # mask ensures that only predicted-stable condensates contribute to the element consumption.
    element_used: Float[Array, "... n_elements"] = jnp.einsum(
        "es,...s->...e", formula_matrix, condensate_stable_mask * n_condensate_fallback
    )
    # jax.debug.print("element_used = {out}", out=element_used)

    # Remaining element budget after stable-condensate allocation, which must be positive or zero.
    remaining_b: Float[Array, "... n_elements"] = jnp.maximum(
        element_abundance - element_used, 0.0
    )
    # jax.debug.print("remaining_b = {out}", out=remaining_b)

    # If present, exclude O2 from the remaining budget, because otherwise the initial estimate for
    # O2 tends to be unreasonably high since oxygen is equally shared among all O-bearing species.
    O2_index: Float[Array, ""] = parameters.reaction_system.phase_system.gas.O2_index
    non_condensate_no_O2_mask: Bool[Array, "... n_species"] = lax.cond(
        jnp.isnan(O2_index),
        lambda: ~condensate_mask,
        lambda: (~condensate_mask).at[O2_index.astype(int)].set(False),  # Additionally turn off O2
    )

    # Allocate remaining budget to non-condensate species excluding O2
    n_other: Float[Array, "... n_species"] = max_moles_by_limiting_element(
        formula_matrix, remaining_b, non_condensate_no_O2_mask
    )
    # NaNs are either O2 (if present) or condensates, both of which should be zeroed for
    # calculating the budget
    n_other = jnp.where(jnp.isnan(n_other), 0.0, n_other)
    # jax.debug.print("n_other = {out}", out=n_other)

    # Recalculate element_used after n_other allocation
    element_used_no_O2: Float[Array, "... n_elements"] = jnp.einsum(
        "es,...s->...e", formula_matrix, n_other
    )
    # jax.debug.print("element_abundance = {out}", out=element_abundance)
    # jax.debug.print("element_used_no_O2 = {out}", out=element_used_no_O2)

    # Update element budget after allocation excluding O2, which must be positive or zero.
    remaining_b = jnp.maximum(element_abundance - element_used - element_used_no_O2, 0.0)
    # jax.debug.print("remaining_b (after n_other) = {out}", out=remaining_b)

    def allocate_to_O2(
        n_other: Float[Array, "... n_species"],
        remaining_b: Float[Array, "... n_elements"],
        formula_matrix: Float[Array, "n_elements n_species"],
        O2_index_int: Integer[Array, ""],
    ) -> Float[Array, "... n_species"]:
        """Allocates remaining O budget to O2 if O2 is present in the system."""
        # NOTE: A sentinel value is required since all branches of lax.cond are traced, but
        # O_index will only ever be meaningful if O2 is present in the system.
        O_index: int = (
            parameters.element_names.index("O") if "O" in parameters.element_names else -1
        )
        # Compute how much O is left
        O_remaining: Float[Array, "..."] = remaining_b[..., O_index]
        # Get stoichiometry of O in O2 (should be 2)
        O2_stoich: Float[Array, ""] = formula_matrix[O_index, O2_index_int]
        n_O2: Float[Array, "..."] = O_remaining / O2_stoich
        # Set the O2 entry in n_other
        n_other = n_other.at[..., O2_index_int].set(n_O2)

        return n_other

    n_other = lax.cond(
        jnp.isnan(O2_index),
        lambda n_other: n_other,
        lambda n_other: allocate_to_O2(n_other, remaining_b, formula_matrix, O2_index.astype(int)),
        n_other,
    )
    # jax.debug.print("n_other after O2 allocation = {out}", out=n_other)

    # Apply fallback logic to non-condensate species.
    # If a species is unconstrained by any available element or constrained to zero moles, fallback
    # values are assigned. For unconstrained species, the fallback value is the geometric mean of
    # the finite estimates, which keeps the magnitude of missing species comparable to known ones.
    # For exhausted elements, the fallback value is a small trace value.
    unconstrained: Bool[Array, "... n_species"] = jnp.isnan(n_other) & ~condensate_mask
    # jax.debug.print("unconstrained = {out}", out=unconstrained)
    exhausted: Bool[Array, "... n_species"] = n_other == 0 & ~condensate_mask
    # jax.debug.print("exhausted = {out}", out=exhausted)
    valid: Bool[Array, "... n_species"] = ~unconstrained & ~exhausted & ~condensate_mask
    # jax.debug.print("valid = {out}", out=valid)

    log_geometric_mean: FloatArray = jnp.sum(
        jnp.where(valid, jnp.log(n_other), 0.0), axis=-1, keepdims=True
    ) / jnp.sum(valid, axis=-1, keepdims=True).astype(float)
    # jax.debug.print("log_geometric_mean = {out}", out=log_geometric_mean)

    n_other_fallback = jnp.where(unconstrained, jnp.exp(log_geometric_mean), n_other)
    n_other_fallback = jnp.where(exhausted, jnp.exp(LOG_TRACE_VALUE), n_other_fallback)

    # jax.debug.print("n_condensate after fallback = {out}", out=n_condensate_fallback)
    # jax.debug.print("n_other after fallback = {out}", out=n_other_fallback)

    return n_condensate_fallback, n_other_fallback


def get_log_activity_estimate(
    parameters: Parameters, species_abundance: Float[Array, "... n_species"]
) -> Float[Array, "... n_species"]:
    """Estimates log activity for gas species based on ideal-gas assumptions.

    For gas species, the log activity is estimated as log(x_i * P) = log(n_i/n_total) + log(P).
    For non-gas species (melt, solid, pure-phase condensates), the log activity is set to 0,
    i.e. unit activity. This ignores dilution (mole fraction < 1) and activity coefficients
    for melt/solid solution species, but is intentional: the pre-screen is a cheap heuristic
    and calling the EOS/mixing models here would be circular and expensive. Activity
    coefficients are in any case unavailable without a complete solution.

    Args:
        parameters: Parameters containing the reaction system and state information
        species_abundance: Estimated abundance of each species

    Returns:
        Estimated log activity for each species
    """
    gas_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.gas_species_mask
    )

    # Must be broadcast to a column array
    pressure: Float[Array, "... 1"] = parameters.state.get_pressure(jnp.log(species_abundance))[
        ..., None
    ]
    n_gas_total: Float[Array, "... 1"] = jnp.nansum(
        jnp.where(gas_mask, species_abundance, 0.0), axis=-1, keepdims=True
    )
    safe_n_gas_total: Float[Array, "... 1"] = jnp.where(n_gas_total > 0, n_gas_total, 1.0)

    # TODO: Here, activity of dissolved species is also computed as unity. To improve.
    log_activity: Float[Array, "... n_species"] = jnp.where(
        gas_mask,
        jnp.log(jnp.where(gas_mask, species_abundance, 1.0))
        - jnp.log(safe_n_gas_total)
        + jnp.log(jnp.where(pressure > 0, pressure, 1.0)),
        0.0,
    )
    # jax.debug.print("log_activity = {out}", out=log_activity)

    # activity: Float[Array, "... n_species"] = jnp.exp(log_activity)
    # jax.debug.print("activity = {out}", out=activity)

    return log_activity


def get_stability_signal(
    parameters: Parameters, species_abundance: Float[Array, "... n_species"]
) -> Float[Array, "... n_reactions n_species"]:
    """Computes the stability signal for each reaction and condensate species.

    The stability signal is defined as sm[r,c] * (log_Kp[r] - log_Q[r]), where sm is the
    stability matrix, Kp is the equilibrium constant, and Q is the reaction quotient. A positive
    signal means the reaction is driven toward condensate formation given the current assumed
    activities.

    Args:
        parameters: Parameters containing the reaction system information
        species_abundance: Estimated abundance of each species, used to estimate activities

    Returns:
        Stability signal for each reaction and condensate species
    """
    reaction_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_matrix_full
    )
    # jax.debug.print("reaction_matrix = {out}", out=reaction_matrix)
    stability_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_stability_matrix_full
    )
    # jax.debug.print("stability_matrix = {out}", out=stability_matrix)

    temperature: FloatArray = parameters.state.temperature
    pressure: FloatArray = parameters.state.get_pressure(jnp.log(species_abundance))
    log_Kp: Float[Array, "... n_reactions"] = parameters.reaction_system.reaction.get_log_Kp(
        temperature, pressure
    )
    # jax.debug.print("log_Kp = {out}", out=log_Kp)

    log_activity: Float[Array, "... n_species"] = get_log_activity_estimate(
        parameters, species_abundance
    )

    # Condensate c is supersaturated when sm[r,c] * (log_Kp[r] - log_Q[r]) > 0.
    log_Q: Float[Array, "... n_reactions"] = jnp.einsum(
        "rs,...s->...r", reaction_matrix, log_activity
    )
    stability_signal: Float[Array, "... n_reactions n_species"] = (
        stability_matrix * jnp.expand_dims(log_Kp - log_Q, axis=-1)
    )
    # jax.debug.print("stability_signal = {out}", out=stability_signal)

    return stability_signal


def generate_auto_initial_guess(parameters: Parameters) -> Float[Array, "... twice_species"]:
    """Generates an automatic initial guess for the solution vector.

    The algorithm:
      - Iteratively predicts stable condensates by allocating element budgets and evaluating
        stability signals until convergence.
      - Allocates element budgets to predicted-stable condensates first, then distributes the
        remainder to other species.
      - Handles fugacity constraints for gas species if present.
      - Initializes log stability for predicted-stable condensates and uses a default value for
        others.

    Args:
        parameters: Parameters for a single batch element

    Returns:
        Concatenated array of [log_number_moles, log_stability]
    """
    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )

    temperature: FloatArray = parameters.state.temperature

    element_abundance: Float[Array, "... n_elements"] = parameters.mass_constraints.abundance()
    # jax.debug.print("element_abundance = {out}", out=element_abundance)

    gas_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.gas_species_mask
    )
    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )
    active_activity_constraints: Bool[Array, "... n_species"] = (
        parameters.activity_constraints.active()
    )
    # jax.debug.print("active_activity_constraints = {out}", out=active_activity_constraints)

    gas_no_imposed_fugacity: Bool[Array, "... n_species"] = gas_mask & ~active_activity_constraints
    # jax.debug.print("gas_no_imposed_fugacity = {out}", out=gas_no_imposed_fugacity)

    def _one_stability_pass(
        condensate_stable_known: Bool[Array, "... n_species"],
    ) -> Bool[Array, "... n_species"]:
        """One stability-prediction pass

        Given a set of already-known-stable condensates, allocates element budget to them first,
        distributes the remainder to gas species, computes ideal activities, evaluates the reaction
        K vs Q signal, and returns the monotone union of the new predictions with the input mask.

        Args:
            condensate_stable_known: Boolean mask of condensates predicted stable so far

        Returns:
            Updated mask - superset of ``condensate_stable_known``.
        """
        _, n_other = allocate_element_budget(
            parameters, element_abundance, condensate_stable_known
        )
        # jax.debug.print("n_other in stability pass = {out}", out=n_other)
        stability_signal: Float[Array, "... n_reactions n_species"] = get_stability_signal(
            parameters, n_other
        )
        # jax.debug.print("stability_signal in stability pass = {out}", out=stability_signal)
        new_predictions: Bool[Array, "... n_species"] = (
            jnp.any(stability_signal > 0, axis=-2) & condensate_mask
        )
        # jax.debug.print("new_predictions in stability pass = {out}", out=new_predictions)
        # jax.debug.print("new_predictions = {out}", out=new_predictions)

        # Monotone union: condensates are never retracted once predicted stable. This ensures the
        # set grows by at least one entry per iteration, guaranteeing fixed-point convergence
        # without a cap. The solver makes the definitive stable/absent decision.
        return condensate_stable_known | new_predictions

    # Iterate until the predicted-stable set stops growing
    # Initialize with no condensates known stable; the first body call is the gas-only pre-screen.
    # The monotone union in _one_stability_pass guarantees termination.
    init_stable: Bool[Array, " n_species"] = jnp.zeros_like(condensate_mask)
    # jax.debug.print("init_stable = {out}", out=init_stable)
    first_stable: Bool[Array, "... n_species"] = _one_stability_pass(init_stable)
    # jax.debug.print("first_stable = {out}", out=first_stable)

    init_stable_broadcasted: Bool[Array, "... n_species"] = jnp.broadcast_to(
        init_stable, first_stable.shape
    )
    # jax.debug.print("init_stable_broadcasted = {out}", out=init_stable_broadcasted)

    def _cond_fn(carry: tuple) -> Bool[Array, "..."]:
        prev, curr, i = carry
        return jnp.any(prev != curr)

    def _body_fn(carry: tuple) -> tuple:
        _, curr, i = carry
        return curr, _one_stability_pass(curr), i + 1

    _, condensate_stable_predicted, _ = lax.while_loop(
        _cond_fn, _body_fn, (init_stable_broadcasted, first_stable, jnp.array(0))
    )

    n_condensate, n_other = allocate_element_budget(
        parameters, element_abundance, condensate_stable_predicted
    )

    # Combine: predicted-stable condensates use their budget estimate; all others use the
    # non-condensate remainder. Fallback for species not covered by any constrained element:
    # geometric mean of finite estimates keeps missing species at a comparable magnitude.
    n_estimate: Float[Array, "... n_species"] = jnp.where(condensate_mask, n_condensate, n_other)
    # jax.debug.print("n_estimate after merge = {out}", out=n_estimate)

    log_number_moles: Float[Array, "... n_species"] = jnp.log(n_estimate)
    # jax.debug.print("log_number_moles = {out}", out=log_number_moles)

    # Fugacity-constrained gas species
    log_n_gas_known_total: Float[Array, "..."] = logsumexp(
        jnp.where(gas_no_imposed_fugacity, log_number_moles, -jnp.inf), axis=-1, keepdims=True
    )
    # jax.debug.print("log_n_gas_known_total = {out}", out=log_n_gas_known_total)

    pressure: Float[Array, "..."] = parameters.state.get_pressure(log_number_moles)
    # jax.debug.print("pressure = {out}", out=pressure)

    # Pressure must be 1-D in this function
    log_fug: Float[Array, "... n_species"] = parameters.activity_constraints.log_activity(
        temperature, pressure
    )
    # jax.debug.print("log_fug = {out}", out=log_fug)

    # Pressure must be a column vector
    log_n_fug: Float[Array, "... n_species"] = (
        log_fug + log_n_gas_known_total - jnp.log(pressure)[..., None]
    )
    # jax.debug.print("log_n_fug = {out}", out=log_n_fug)

    log_number_moles = jnp.where(
        gas_mask & active_activity_constraints, log_n_fug, log_number_moles
    )
    # jax.debug.print("log_number_moles after fugacity constraints = {out}", out=log_number_moles)

    log_stability = compute_implied_log_stability(parameters, log_number_moles)
    # jax.debug.print("log_stability = {out}", out=log_stability

    result: Float[Array, "... twice_species"] = jnp.concatenate(
        (log_number_moles, log_stability), axis=-1
    )
    # jax.debug.print("Initial guess (log_number_moles, log_stability) = {out}", out=result)

    result = eqx.error_if(
        result,
        jnp.any(jnp.isnan(result)),
        "Initial guess contains NaNs, which will break the solver.",
    )

    return result


class HeuristicInitializer(eqx.Module):
    """Initializer that uses the heuristic guess from :func:`generate_auto_initial_guess`"""

    def start(self, parameters: Parameters) -> Float[Array, " twice_species"]:
        """Cold initial guess from :func:`generate_auto_initial_guess`.

        Args:
            parameters: Parameters of a single system

        Returns:
            Concatenated array of [log_number_moles, log_stability]
        """
        return generate_auto_initial_guess(parameters)


def _gibbs_over_RT_all_species(
    parameters: Parameters, temperature: Array, pressure: Array
) -> Float[Array, " n_species"]:
    """Standard Gibbs energy over RT for every species, zero for species not in the reactions

    Args:
        parameters: Parameters of a single system
        temperature: Temperature (K)
        pressure: Pressure (bar)

    Returns:
        Standard Gibbs energy over RT in the full species space
    """
    species = parameters.reaction_system.species
    reaction_names: list[str] = list(species.reaction_species.species_names)
    gibbs_reaction: Float[Array, " n_reaction_species"] = (
        parameters.reaction_system.reaction.vmap_gibbs(
            jnp.arange(len(reaction_names)), temperature, pressure
        )
    )

    return jnp.stack(
        [
            gibbs_reaction[reaction_names.index(name)] if name in reaction_names else 0.0
            for name in species.species_names
        ]
    )


class ElementPotentialInitializer(eqx.Module):
    r"""Initializer that solves an ideal-gas equilibrium in the element potentials

    The cold guess is the solution of a simplified equilibrium problem: ideal gas, pure
    condensates and standard Gibbs energies held fixed, solved for the element potentials. This
    is the default initializer of :class:`~atmodeller.classes.EquilibriumModel`.

    Notation: :math:`\lambda_e` is the potential of element :math:`e` in units of :math:`RT`,
    :math:`a_{es}` the formula matrix (moles of element :math:`e` in species :math:`s`),
    :math:`G_s/RT` the standard Gibbs energy of species :math:`s`, :math:`b_e` the element
    budget (moles), :math:`n_s` and :math:`p_s` the moles and partial pressure of species
    :math:`s`, :math:`N_\mathrm{gas}` the total gas moles, and
    :math:`d_c = \sum_e a_{ec} \lambda_e - G_c/RT` the driving force of condensate :math:`c`
    (positive if supersaturated).

    .. rubric:: Model

    - Gas: ideal, :math:`\ln p_s = \sum_e a_{es} \lambda_e - G_s/RT` and
      :math:`n_s = (p_s / \sum_\mathrm{gas} p)\, N_\mathrm{gas}`.
    - Condensates: pure phases; a present condensate has unit activity, :math:`d_c = 0`.
    - Dissolved species (if the system has dissolution reactions): their log moles are
      unknowns, and the dissolution rows of the solver's residual (the solubility laws) are part
      of the residual.
    - :math:`G_s/RT` is evaluated once, at the pressure of the start, and held fixed.

    The unknowns are :math:`z = [\lambda, \ln N_\mathrm{gas}, \text{one slot per species}]`,
    where the slots hold :math:`\ln n` of present condensates and dissolved species (the others
    are unused). The residual has a fixed shape, with inapplicable rows zero:

    .. math::

        \begin{array}{ll}
        \ln \left( \sum_s a_{es} n_s / b_e \right) & \text{mass balance, constrained elements} \\
        \ln p_s - \ln f_s & \text{imposed gas fugacities} \\
        d_c & \text{present condensates} \\
        \ln n_c - \texttt{LOG\_TRACE\_VALUE} & \text{absent condensate slots} \\
        \ln \sum_\mathrm{gas} p - \ln P(n) & \text{pressure closure} \\
        \text{dissolution residual} & \text{dissolved species}
        \end{array}

    The mass balance sums gas, present condensates and dissolved species. The pressure closure
    and imposed fugacities are needed: the pressure of a planet depends on its atmosphere, and
    an element that is not mass constrained (e.g. O under an fO\ :sub:`2` buffer) has its
    potential fixed only by an imposed fugacity.

    .. rubric:: Method

    1. Start from an even split of the total constrained element moles over the gas species (as
       CEA), with :math:`\lambda` fitted to it by least squares. No knowledge of the answer is
       used.
    2. Phase selection in :math:`2 n_\mathrm{condensates} + 1` passes, then a final solve. Each
       pass solves, switches on the most supersaturated absent condensate (if
       :math:`d_c > 10^{-6}`; it starts at half the budget of its limiting element), and
       switches off present condensates whose log moles fall below :data:`LOG_TRACE_VALUE` + 1.
       Switching one at a time means that a present polymorph fixes its element's potential and
       the other becomes undersaturated.
    3. Every solve is an element sweep followed by damped Newton:

       - *Sweep* (FastChem): ``sweeps`` Gauss-Seidel passes over the elements in order of
         decreasing abundance, each solving its own element's mass balance exactly for
         :math:`\lambda_e` (see below).
       - *Newton*: least-squares direction (``lstsq`` of the Jacobian, since the fixed-shape
         system has zero rows), scaled so that no :math:`\lambda_e` and no carried species' log
         moles change by more than ``max_log_step``; stops when the residual norm is below
         ``residual_tolerance``, or after ``newton_iterations``.

    4. Absent condensates are given the solver's stability representation: the solver's
       residual for an absent condensate is :math:`-\nu (d_c + \zeta_c)`, so
       :math:`\zeta_c = -d_c`, and its log moles follow from the implied-stability relation
       (absent condensates with :math:`d_c \geq 0` stay at trace).
    5. If the result is not finite, the even-split start is returned instead.

    .. rubric:: Why the sweep

    In the element-potential form a gas species is not an unknown. It enters Newton only
    through the mass-balance rows, with a sensitivity proportional to its share of the element
    budget. If it is driven far below the major carriers of its elements, that sensitivity
    underflows and Newton cannot bring it back. This is the *water trap*: with liquid water
    present and hydrogen slightly in excess of H\ :sub:`2`\ O stoichiometry, the excess
    hydrogen must be H\ :sub:`2` gas, but a start with H\ :sub:`2` at
    :math:`\ln n \approx -18` next to H\ :sub:`2`\ O\ :sub:`l` at 52.7 leaves the hydrogen
    balance only a few percent off, with a derivative of about :math:`e^{-70}` in the
    H\ :sub:`2` direction.

    FastChem (:cite:p:`Stock2018,Stock2022,KSP24`) solves for the same unknowns (its log
    free-atom densities differ from :math:`\lambda_e` by a constant) but by Gauss-Seidel over
    the elements: each element's own mass balance, solved on its own scale, so a minor species
    is found however small it starts. Gauss-Seidel alone converges slowly when elements are
    strongly coupled and knows nothing of condensates, fugacities or the pressure closure, so
    here it only sets the scale of every species, and Newton then solves the coupled problem.

    *The 1-D problem for element e.* Hold the other potentials, the moles held outside the gas
    (:math:`C_e`, in present condensates and dissolved species) and the gas normalisation
    :math:`\sigma = \ln N_\mathrm{gas} - \ln \sum_\mathrm{gas} p` fixed, so that
    :math:`\ln n_s = a_{es} x + \beta_s` with :math:`x = \lambda_e`. The log mass balance

    .. math::

        f(x) = \ln \sum_{s \in \mathrm{gas},\, a_{es} > 0}
               \exp\left(\ln a_{es} + a_{es} x + \beta_s\right) - \ln(b_e - C_e)

    has :math:`f'(x) = \sum_s w_s a_{es} \geq 1` and :math:`f''(x) \geq 0` (mean and variance
    of :math:`a_{es}` under the weights :math:`w_s = a_{es} n_s / \sum a n`). It is convex and
    increasing with slope at least one, so it has one root and Newton reaches it from any
    start, with every step at most :math:`|f|`. Freezing :math:`\sigma` is FastChem's fixed
    total density: it keeps each 1-D problem separate and solvable when :math:`b_e > C_e`.
    After the sweep :math:`\ln N_\mathrm{gas}` is reset so that the normalised moles equal the
    swept ones. An element is skipped if it is not mass constrained, if :math:`C_e \geq b_e`, or
    if no gas species contains it. The sweep ignores every row other than the mass balances.

    .. rubric:: JAX

    The guess is computed per system and vmapped over the batch by the solver. The sweep, the switching passes and the 1-D Newton iterations have fixed counts
    (``lax.fori_loop``); each Newton solve is a ``lax.while_loop``, which under vmap runs until
    every system in the batch has converged (or hit the cap). The element order comes from
    ``argsort`` of the (possibly traced) abundances. Gradients are stopped through the guess,
    since the equilibrium solve is differentiated implicitly (and ``while_loop`` is not
    reverse-differentiable).

    Args:
        newton_iterations: Maximum number of damped Newton iterations per solve. Defaults to
            ``100``.
        residual_tolerance: Residual norm at which a Newton solve stops. Defaults to ``1e-10``.
        max_log_step: Maximum change in the log moles of any carried species (or in any
            :math:`\lambda_e`) per Newton iteration. Defaults to ``2.0``.
        sweeps: Gauss-Seidel element sweeps at the start of every Newton solve. Defaults to
            ``3``.
        sweep_iterations: 1-D Newton iterations per element in a sweep. Defaults to ``30``.
    """

    newton_iterations: int = eqx.field(static=True, default=100)
    """Maximum number of damped Newton iterations per solve"""
    residual_tolerance: float = eqx.field(static=True, default=1e-10)
    """Residual norm at which a Newton solve stops"""
    max_log_step: float = eqx.field(static=True, default=2.0)
    r"""Maximum change in the log moles of any carried species (or in any :math:`\lambda_e`) per
    Newton iteration"""
    sweeps: int = eqx.field(static=True, default=3)
    """Gauss-Seidel element sweeps at the start of every Newton solve"""
    sweep_iterations: int = eqx.field(static=True, default=30)
    """1-D Newton iterations per element in a sweep (each step is at most the log residual)"""

    def start(self, parameters: Parameters) -> Float[Array, " twice_species"]:
        """Cold initial guess from the element-potential equilibrium.

        Args:
            parameters: Parameters of a single system

        Returns:
            Concatenated array of [log_number_moles, log_stability]
        """
        # The guess needs no gradients (the solve is differentiated implicitly), and reverse-mode
        # differentiation cannot pass through the while loops
        parameters = jax.tree_util.tree_map(
            lambda x: lax.stop_gradient(x) if eqx.is_inexact_array(x) else x, parameters
        )

        # -----------------------------------------------------------------------------------
        # STEP 1. Setup: species classes, formula matrix, element budgets, constraints.
        #
        # Gas species and condensates are in the reaction network; dissolved species are not
        # (they have no standard Gibbs energy) and are handled through the dissolution rows.
        # Unconstrained elements have a NaN budget.
        # -----------------------------------------------------------------------------------
        reaction_system = parameters.reaction_system
        species = reaction_system.species
        reaction_names: list[str] = list(species.reaction_species.species_names)
        n_species: int = species.number_species
        formula_matrix: Float[Array, "n_elements n_species"] = jnp.asarray(
            reaction_system.formula_matrix, dtype=float
        )
        n_elements: int = formula_matrix.shape[0]
        in_reaction: np.ndarray = np.array(
            [name in reaction_names for name in species.species_names]
        )
        gas: Bool[Array, " n_species"] = (
            jnp.asarray(reaction_system.phase_system.gas_species_mask) & in_reaction
        )
        condensate: Bool[Array, " n_species"] = (
            jnp.asarray(reaction_system.phase_system.condensates_species_mask) & in_reaction
        )
        n_core: int = reaction_system.reaction.number_reactions
        has_dissolution: bool = reaction_system.dissolution.number_reactions > 0
        dissolved: Bool[Array, " n_species"] = jnp.asarray(~in_reaction & has_dissolution)
        temperature: Array = jnp.asarray(parameters.state.temperature)
        abundance: Float[Array, " n_elements"] = jnp.reshape(
            parameters.mass_constraints.abundance(), (n_elements,)
        )
        constrained: Bool[Array, " n_elements"] = ~jnp.isnan(abundance) & (abundance > 0)
        log_b: Float[Array, " n_elements"] = jnp.log(jnp.where(constrained, abundance, 1.0))
        imposed: Bool[Array, " n_species"] = gas & parameters.activity_constraints.active()

        # -----------------------------------------------------------------------------------
        # STEP 2. Even-split start and standard Gibbs energies (at the start pressure, held
        # fixed).
        # -----------------------------------------------------------------------------------
        total: Array = jnp.sum(jnp.where(constrained, abundance, 0.0))
        total = jnp.where(total > 0, total, 1.0)
        log_n_each: Array = jnp.log(total / jnp.maximum(jnp.sum(gas), 1))
        start_log_n: Float[Array, " n_species"] = jnp.where(gas, log_n_each, LOG_TRACE_VALUE)
        gibbs: Float[Array, " n_species"] = _gibbs_over_RT_all_species(
            parameters, temperature, parameters.state.get_pressure(start_log_n)
        )

        # -----------------------------------------------------------------------------------
        # STEP 3. Species moles from the unknowns z = [lambda, ln N_gas, slots].
        #
        # Gas from the element potentials (normalised to N_gas); present condensates and
        # dissolved species from their slots; absent condensates at trace.
        # -----------------------------------------------------------------------------------
        def log_moles(z: Array, present: Array) -> tuple[Array, Array, Array]:
            lam, log_n_gas, slots = z[:n_elements], z[n_elements], z[n_elements + 1 :]
            log_p: Array = jnp.clip(formula_matrix.T @ lam - gibbs, -300.0, 300.0)
            log_p_sum: Array = logsumexp(jnp.where(gas, log_p, -jnp.inf))
            log_n: Array = jnp.where(
                gas,
                log_p - log_p_sum + log_n_gas,
                jnp.where((condensate & present) | dissolved, slots, LOG_TRACE_VALUE),
            )
            return jnp.clip(log_n, -300.0, 300.0), log_p, log_p_sum

        # -----------------------------------------------------------------------------------
        # STEP 4. Residual (fixed shape; inapplicable rows zero). See the class docstring.
        #
        # The mass balance is a log ratio, so every element weighs the same whatever its
        # abundance. Fugacity targets may depend on pressure.
        # -----------------------------------------------------------------------------------
        def residual(z: Array, present: Array) -> Array:
            lam, slots = z[:n_elements], z[n_elements + 1 :]
            log_n, log_p, log_p_sum = log_moles(z, present)
            pressure: Array = parameters.state.get_pressure(log_n)
            carried: Array = gas | (condensate & present) | dissolved
            element_moles: Array = formula_matrix @ jnp.where(carried, jnp.exp(log_n), 0.0)
            mass: Array = jnp.where(
                constrained, jnp.log(jnp.where(constrained, element_moles, 1.0)) - log_b, 0.0
            )
            target: Array = parameters.activity_constraints.log_activity(temperature, pressure)
            fugacity: Array = jnp.where(imposed, log_p - jnp.where(imposed, target, 0.0), 0.0)
            condensed: Array = jnp.where(
                condensate & present,
                formula_matrix.T @ lam - gibbs,
                jnp.where(condensate, slots - LOG_TRACE_VALUE, 0.0),
            )
            closure: Array = jnp.atleast_1d(log_p_sum - jnp.log(pressure))
            rows: list[Array] = [mass, fugacity, condensed, closure]
            if has_dissolution:
                # The solver's dissolution rows, which follow its core reaction rows
                rows.append(
                    reaction_system.get_residual(
                        log_n,
                        reaction_system.get_log_activity(log_n, temperature, pressure),
                        compute_implied_log_stability(parameters, log_n),
                        temperature,
                        pressure,
                    )[n_core:]
                )
            return jnp.concatenate(rows)

        # -----------------------------------------------------------------------------------
        # STEP 5. Element sweep (FastChem): each element's own mass balance, solved for
        # lambda_e.
        #
        # See the class docstring for the 1-D problem. Held fixed: the other potentials (updated
        # in Gauss-Seidel order, most abundant element first), the moles outside the gas (C_e)
        # and the gas normalisation sigma.
        # -----------------------------------------------------------------------------------
        order: Array = jnp.argsort(-jnp.where(constrained, abundance, -jnp.inf))
        # ln a_es; the placeholder 1.0 where a_es = 0 is masked out by the carrier mask
        log_a: Array = jnp.log(jnp.where(formula_matrix > 0, formula_matrix, 1.0))

        def sweep(z: Array, present: Array) -> Array:
            log_n, _, log_p_sum = log_moles(z, present)
            sigma: Array = z[n_elements] - log_p_sum
            outside_gas: Array = formula_matrix @ jnp.where(
                (condensate & present) | dissolved, jnp.exp(log_n), 0.0
            )

            def element(k, lam):
                e: Array = order[k]
                a_e: Array = formula_matrix[e]
                carrier: Array = gas & (a_e > 0)
                gas_share: Array = abundance[e] - outside_gas[e]
                solvable: Array = constrained[e] & (gas_share > 0) & jnp.any(carrier)
                log_target: Array = jnp.log(jnp.where(solvable, gas_share, 1.0))
                # beta_s: ln n_s without its a_es lambda_e term
                beta: Array = formula_matrix.T @ lam - gibbs + sigma - a_e * lam[e]

                def newton(_, x):
                    terms: Array = jnp.where(carrier, log_a[e] + a_e * x + beta, -jnp.inf)
                    log_g: Array = logsumexp(terms)
                    # f' = mean stoichiometry of the carriers, >= 1 (floor for non-integer
                    # formulae)
                    slope: Array = jnp.maximum(jnp.sum(jnp.exp(terms - log_g) * a_e), 1.0)
                    # Clipped only as a guard against extreme values
                    return x - jnp.clip((log_g - log_target) / slope, -50.0, 50.0)

                x: Array = lax.fori_loop(0, self.sweep_iterations, newton, lam[e])
                return lam.at[e].set(jnp.where(solvable & jnp.isfinite(x), x, lam[e]))

            lam: Array = lax.fori_loop(
                0,
                self.sweeps,
                lambda _, lam: lax.fori_loop(0, n_elements, element, lam),
                z[:n_elements],
            )
            # Reset ln N_gas so that the normalised gas moles of STEP 3 equal the swept ones
            log_p_sum_new: Array = logsumexp(
                jnp.where(gas, jnp.clip(formula_matrix.T @ lam - gibbs, -300.0, 300.0), -jnp.inf)
            )
            return z.at[:n_elements].set(lam).at[n_elements].set(log_p_sum_new + sigma)

        # -----------------------------------------------------------------------------------
        # STEP 6. Solve: sweep, then damped Newton to residual_tolerance (at most
        # newton_iterations).
        #
        # Step control: no lambda and no carried species' log moles (via a jvp through STEP 3)
        # change by more than max_log_step. Without it, Newton on exponentials overshoots minor
        # species into flat regions where it stalls.
        # -----------------------------------------------------------------------------------
        def solve(z: Array, present: Array) -> Array:
            carried: Array = gas | (condensate & present) | dissolved

            def newton(carry):
                k, z, r = carry
                dz: Array = jnp.linalg.lstsq(jax.jacfwd(residual)(z, present), -r)[0]
                dz = jnp.where(jnp.isfinite(dz), dz, 0.0)
                _, dlog_n = jax.jvp(lambda z_: log_moles(z_, present)[0], (z,), (dz,))
                biggest: Array = jnp.maximum(
                    jnp.max(jnp.where(carried, jnp.abs(dlog_n), 0.0)),
                    jnp.max(jnp.abs(dz[:n_elements])),
                )
                scale: Array = jnp.minimum(1.0, self.max_log_step / jnp.maximum(biggest, 1e-12))
                z = z + scale * dz
                return k + 1, z, residual(z, present)

            def unfinished(carry):
                k, _, r = carry
                return (k < self.newton_iterations) & (
                    jnp.linalg.norm(r) > self.residual_tolerance
                )

            z = sweep(z, present)
            return lax.while_loop(unfinished, newton, (jnp.array(0), z, residual(z, present)))[1]

        # -----------------------------------------------------------------------------------
        # STEP 7. Phase selection: switch condensates on and off one at a time.
        #
        # Starting unknowns: lambda fitted to the even split by least squares
        # (ln n_s = a_s.lambda - G_s/RT + c over the gas species), ln N_gas = ln(total),
        # dissolved slots at the even split, condensate slots at trace.
        # 2 * n_condensates + 1 passes let each condensate be switched on and off once; a final
        # solve follows.
        # -----------------------------------------------------------------------------------
        design: Array = jnp.where(
            gas[:, None], jnp.concatenate([formula_matrix.T, jnp.ones((n_species, 1))], -1), 0.0
        )
        fit: Array = jnp.linalg.lstsq(design, jnp.where(gas, log_n_each + gibbs, 0.0))[0]
        z0: Array = jnp.concatenate(
            [
                fit[:n_elements],
                jnp.atleast_1d(jnp.log(total)),
                jnp.where(dissolved, log_n_each, LOG_TRACE_VALUE),
            ]
        )

        def switching_pass(_, carry):
            z, present = carry
            z = solve(z, present)
            slots: Array = z[n_elements + 1 :]
            driving_force: Array = formula_matrix.T @ z[:n_elements] - gibbs
            candidates: Array = condensate & ~present & (driving_force > 1e-6)
            best: Array = jnp.argmax(jnp.where(candidates, driving_force, -jnp.inf))
            on: Array = (jnp.arange(n_species) == best) & jnp.any(candidates)
            off: Array = condensate & present & (slots < LOG_TRACE_VALUE + 1.0)
            # A newly present condensate starts at half the budget of its limiting element
            n_limit: Array = max_moles_by_limiting_element(
                formula_matrix, jnp.where(constrained, abundance, jnp.nan), on
            )
            log_start: Array = jnp.log(
                0.5 * jnp.where(n_limit > 0, n_limit, jnp.exp(LOG_TRACE_VALUE))
            )
            slots = jnp.where(on, log_start, jnp.where(off, LOG_TRACE_VALUE, slots))
            return z.at[n_elements + 1 :].set(slots), (present | on) & ~off

        n_condensates: int = len(reaction_names) - len(reaction_system.phase_system.gas.species)
        z, present = lax.fori_loop(
            0, 2 * n_condensates + 1, switching_pass, (z0, jnp.zeros(n_species, dtype=bool))
        )
        z = solve(z, present)

        # -----------------------------------------------------------------------------------
        # STEP 8. Output: absent condensates in the solver's stability representation
        # (zeta_c = -d_c), then [log moles, log stability], or the even-split start if not
        # finite.
        # -----------------------------------------------------------------------------------
        log_n: Array = log_moles(z, present)[0]
        driving_force: Array = formula_matrix.T @ z[:n_elements] - gibbs
        absent: Array = condensate & ~present & (driving_force < 0)
        implied_constant: Array = compute_implied_log_stability(parameters, jnp.zeros(n_species))
        log_n = jnp.where(
            absent, implied_constant - jnp.log(jnp.where(absent, -driving_force, 1.0)), log_n
        )
        log_n = jnp.where(in_reaction | dissolved, log_n, LOG_TRACE_VALUE)
        result: Array = jnp.concatenate([log_n, compute_implied_log_stability(parameters, log_n)])
        start: Array = jnp.concatenate(
            [start_log_n, compute_implied_log_stability(parameters, start_log_n)]
        )

        return jnp.where(jnp.all(jnp.isfinite(result)), result, start)
