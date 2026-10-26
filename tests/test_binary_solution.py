# SPDX-License-Identifier: Apache-2.0

"""Native solver integration and phase-output checks for binary demixing."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atmodeller import (
    ActivityCoefficient,
    BinarySolutionPhase,
    ChemicalSpecies,
    CondensedPhase,
    EquilibriumModel,
    GasPhase,
    SolverParameters,
    SubregularSolution,
    ThermodynamicState,
)
from atmodeller.phases import BinarySolutionPhaseOutput
from atmodeller.reactions import PhaseSystem, ReactionSystem
from atmodeller.sci_utils import GAS_CONSTANT as R


def make_phase(interaction=3 * R * 2000, *, equilibrate=True):
    components = tuple(ChemicalSpecies.create_solution(s) for s in ("Fe", "Si"))
    return BinarySolutionPhase(
        components,
        SubregularSolution.regular_solution(interaction),
        "metal",
        equilibrate=equilibrate,
    )


@pytest.fixture(scope="module", params=["ideal", "homogeneous", "demixed"])
def equilibrium(request):
    mode = request.param
    binary = make_phase(0 if mode == "ideal" else 3 * R * 2000, equilibrate=mode != "homogeneous")
    gas = GasPhase([ChemicalSpecies.create_gas(s) for s in ("He", "Fe", "Si")])
    phases = PhaseSystem(
        gas,
        other_phases=(
            CondensedPhase(name="silicate_melt"),
            CondensedPhase(name="silicate_solid"),
            binary,
        ),
    )
    state = ThermodynamicState(ReactionSystem(phases), 2000.0, 1.0)
    masses = {
        "He": 0.004002602,
        "Fe": 0.75 * binary.species[0].data.molar_mass,
        "Si": 0.25 * binary.species[1].data.molar_mass,
    }
    model = EquilibriumModel.from_state(
        state, mass_constraints=masses, solver_parameters=SolverParameters(atol=1e-10, rtol=1e-10)
    )
    output = model.solve_with_default()
    assert np.asarray(output.multi_attempt_solution.success).all()
    return mode, binary, gas, masses, output


def test_native_equilibrium_conserves_every_element(equilibrium):
    _, _, _, masses, output = equilibrium
    d = output.to_dict(to_numpy=True)
    for element, target in masses.items():
        total = sum(
            np.asarray(d[p]["elements"]["mass"].get(element, 0)).sum() for p in ("gas", "metal")
        )
        assert total == pytest.approx(target, rel=1e-10)
    pressures = d["gas"]["species"]["partial_pressure"]
    assert sum(float(np.asarray(v).item()) for v in pressures.values()) == pytest.approx(
        1, abs=1e-12
    )


def test_vapor_pressures_match_independent_chemical_potential_balance(equilibrium):
    mode, binary, gas, _, output = equilibrium
    d = output.to_dict(to_numpy=True)
    activities = d["metal"]["species"]["activity"]
    pressures = d["gas"]["species"]["partial_pressure"]
    for liquid in binary.species:
        vapor = next(s for s in gas.species if s.data.formula == liquid.data.formula)
        delta = liquid.thermo.get_gibbs_over_RT(2000.0, 1.0) - vapor.thermo.get_gibbs_over_RT(
            2000.0, 1.0
        )
        expected = np.exp(delta) * activities[liquid.name]
        np.testing.assert_allclose(pressures[vapor.name], expected, rtol=1e-10)
    if mode == "demixed":
        # Symmetry fixes equal component activities independently of the bulk
        # composition. Gas composition follows the pure-standard-state difference.
        np.testing.assert_allclose(activities["Fe_l"], activities["Si_l"], rtol=1e-12)
        np.testing.assert_allclose(pressures["Si_g"], 4.105485632847144e-5, rtol=1e-9)
    elif mode == "ideal":
        np.testing.assert_allclose(pressures["Si_g"], 1.088295776309555e-5, rtol=1e-9)


@pytest.mark.parametrize("output_format", ["natural", "named_arrays", "elements_species"])
def test_stability_and_lever_rule_survive_output_formats(equilibrium, output_format):
    mode, _, _, _, output = equilibrium
    d = output.to_dict(output_format=output_format, to_numpy=True)
    diagnostic = d["metal"]["phase"]["mixing"]
    assert diagnostic["valid"].shape == (1, 1)
    assert bool(diagnostic["valid"].item())
    assert bool(diagnostic["demixing_enabled"].item()) == (mode != "homogeneous")
    assert bool(diagnostic["spinodally_unstable"].item()) == (mode != "ideal")
    if mode != "ideal":
        f = diagnostic["equilibrium_upper_phase_mole_fraction"]
        reconstructed = (1 - f) * diagnostic["binodal_lower"] + f * diagnostic["binodal_upper"]
        np.testing.assert_allclose(
            reconstructed, diagnostic["bulk_component_1_mole_fraction"], rtol=1e-13
        )


def test_phase_batch_and_trace_limits():
    phase = make_phase()
    logs = jnp.log(jnp.array([[0.7, 0.3], [0.999, 0.001]]))
    temperatures = jnp.array([2000.0, 5000.0])
    activities = eqx.filter_jit(phase.get_log_activity)(logs, temperatures, jnp.array(1.0))
    np.testing.assert_allclose(activities[0, 0], activities[0, 1], atol=1e-13)
    np.testing.assert_allclose(
        activities[1], phase.mixing_model.log_activities(5000.0, 1.0, 0.001)
    )
    out = phase.output(logs, jnp.full_like(logs, -jnp.inf), temperatures, jnp.array(1.0))
    assert isinstance(out, BinarySolutionPhaseOutput)
    assert all(v.shape == (2, 1) for v in out.mixing.values())
    trace = phase.get_log_activity(jnp.array([0.0, -500.0]), jnp.array(2000.0), jnp.array(1.0))
    assert np.isfinite(trace).all()
    assert float(trace[1]) == pytest.approx(-497.0)
    # Amount derivatives inside the gap vanish: changing bulk composition only
    # changes the lever-rule amounts, not the shared component activities.
    jacobian = jax.jacfwd(lambda n: phase.get_log_activity(n, jnp.array(2000.0), jnp.array(1.0)))(
        logs[0]
    )
    np.testing.assert_allclose(jacobian, 0, atol=1e-13)


def test_binary_phase_rejects_inconsistent_component_configuration():
    components = [ChemicalSpecies.create_solution(s) for s in ("Fe", "Si")]
    model = SubregularSolution.regular_solution(50000)
    for bad in ([], components[:1], components + components[:1], [components[0], components[0]]):
        with pytest.raises(ValueError):
            BinarySolutionPhase(bad, model)
    for modification in (
        {"include_in_phase_mass": False},
        {"solve_for_stability": True},
        {"number_solution": 0},
        {"activity": ActivityCoefficient(2)},
    ):
        with pytest.raises(ValueError):
            BinarySolutionPhase([replace(components[0], **modification), components[1]], model)
    with pytest.raises(ValueError):
        BinarySolutionPhase([ChemicalSpecies.create_gas("Fe"), components[1]], model)
    for state in ("g", "d"):
        with pytest.raises(ValueError):
            ChemicalSpecies.create_solution("Fe", state=state)


def test_failed_binodal_does_not_fall_back_to_homogeneous_phase():
    phase = make_phase(800 * R * 2000)
    values = phase.get_log_activity(
        jnp.log(jnp.array([0.7, 0.3])), jnp.array(2000.0), jnp.array(1.0)
    )
    assert np.isnan(values).all()


def test_native_batch_crosses_from_demixed_to_miscible():
    components = tuple(ChemicalSpecies.create_solution(s) for s in ("Fe", "Si"))
    omega = 2.3 * R * 2000
    binary = BinarySolutionPhase(
        components, SubregularSolution(omega, omega, pressure_scale=10), "metal"
    )
    gas = GasPhase([ChemicalSpecies.create_gas(s) for s in ("He", "Fe", "Si")])
    phases = PhaseSystem(
        gas,
        other_phases=(
            CondensedPhase(name="silicate_melt"),
            CondensedPhase(name="silicate_solid"),
            binary,
        ),
    )
    temperature = jnp.array([1800.0, 2000.0, 2600.0])
    pressure = jnp.array([1.0, 2.0, 1.0])
    state = ThermodynamicState(ReactionSystem(phases), temperature, pressure)
    masses = {
        "He": 0.004002602,
        "Fe": 0.75 * components[0].data.molar_mass,
        "Si": 0.25 * components[1].data.molar_mass,
    }
    output = EquilibriumModel.from_state(state, mass_constraints=masses).solve_with_default()
    assert np.asarray(output.multi_attempt_solution.success).all()
    d = output.to_dict(to_numpy=True)
    diagnostics = d["metal"]["phase"]["mixing"]
    np.testing.assert_array_equal(
        diagnostics["equilibrium_two_phase"].ravel(), [True, True, False]
    )
    for element, target in masses.items():
        total = sum(np.asarray(d[p]["elements"]["mass"].get(element, 0)) for p in ("gas", "metal"))
        np.testing.assert_allclose(total, target, rtol=1e-8)
    frames = output.to_dataframes()
    assert frames["metal"].shape[0] == 3
    assert "phase.mixing.equilibrium_two_phase" in frames["metal"].columns
