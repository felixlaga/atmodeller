# SPDX-License-Identifier: Apache-2.0

"""Public mode selection, matched controls and physically conserved subphase outputs."""

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from atmodeller import (
    BinarySolutionPhase,
    ChemicalSpecies,
    EquilibriumModel,
    PhaseEquilibriumMode,
    Planet,
    SolverParameters,
    SubregularSolution,
    ThermodynamicState,
)
from atmodeller.sci_utils import GAS_CONSTANT as R
from atmodeller.phases import BinarySolutionPhaseOutput
from atmodeller.state import BasePlanet, BaseThermodynamicState
from atmodeller.thermodata.defaults import default_properties


def components():
    return tuple(ChemicalSpecies.create_solution(s) for s in ("Fe", "Si"))


def gases():
    return tuple(ChemicalSpecies.create_gas(s) for s in ("He", "Fe", "Si"))


def metal(state: BaseThermodynamicState) -> BinarySolutionPhase:
    phase = state.phase_system.phases[3]
    assert isinstance(phase, BinarySolutionPhase)
    return phase


def make_model(mode: PhaseEquilibriumMode = "spinodal", temperature=2000.0):
    return EquilibriumModel.from_species(
        gases(),
        pressure=1.0,
        temperature=temperature,
        mass=0,
        metal_species=components(),
        mixing_models={"metal": SubregularSolution.regular_solution(3 * R * 2000)},
        phase_equilibrium=mode,
        mass_constraints={"He": 1.0, "Fe": 0.75, "Si": 0.25},
        mass_units="moles",
        solver_parameters=SolverParameters(atol=1e-10, rtol=1e-10),
    )


def test_selection_round_trip_and_combined_temperature_update():
    original = make_model()
    homogeneous = original.with_phase_equilibrium("homogeneous")
    back = homogeneous.with_phase_equilibrium("spinodal")
    out = [m.solve_with_default() for m in (original, homogeneous, back)]
    assert all(np.all(o.multi_attempt_solution.success) for o in out)
    pressures = [
        o.to_dict(to_numpy=True)["gas"]["species"]["partial_pressure"]["Si_g"].item() for o in out
    ]
    assert pressures[0] == pytest.approx(pressures[2], rel=1e-12)
    assert pressures[1] / pressures[0] > 1.4
    assert metal(original.parameters.state).equilibrate
    assert not metal(homogeneous.parameters.state).equilibrate
    updated = original.update_state(temperature=2100.0, phase_equilibrium="homogeneous")
    fresh = make_model("homogeneous", 2100.0)
    np.testing.assert_allclose(
        updated.solve_with_default().solution, fresh.solve_with_default().solution, atol=1e-11
    )
    assert original.parameters.state.temperature.item() == 2000
    assert updated.solver is not original.solver


@pytest.mark.parametrize("constructor", [ThermodynamicState, Planet])
def test_state_factories_and_from_state_selector(constructor):
    extra: dict[str, Any] = (
        {"mass": 0} if constructor is ThermodynamicState else {"core_mass_fraction": None}
    )
    state = constructor.from_species(
        gases(),
        pressure=1.0,
        temperature=2000.0,
        metal_species=components(),
        mixing_models={"metal": SubregularSolution.regular_solution(3 * R * 2000)},
        phase_equilibrium="homogeneous",
        **extra,
    )
    switched = state.update(temperature=2100.0, phase_equilibrium="spinodal")
    assert not metal(state).equilibrate
    assert metal(switched).equilibrate
    assert switched.temperature.item() == 2100
    model = EquilibriumModel.from_state(
        state,
        mass_constraints={"He": 1.0, "Fe": 0.75, "Si": 0.25},
        mass_units="moles",
        phase_equilibrium="spinodal",
    )
    assert metal(model.parameters.state).equilibrate
    assert np.all(model.solve_with_default().multi_attempt_solution.success)
    if isinstance(state, BasePlanet):
        with pytest.raises(ValueError, match="background masses"):
            state.update(planet_mass=1e24)


def test_mode_only_update_and_default_compatibility():
    state = make_model().parameters.state
    assert not metal(state.update(phase_equilibrium="homogeneous")).equilibrate
    assert state.update() is state
    ordinary = ThermodynamicState.from_species(gases(), pressure=1.0)
    ordinary_homogeneous = ordinary.with_phase_equilibrium("homogeneous")
    assert (
        ordinary_homogeneous.phase_system.species.species_names
        == ordinary.phase_system.species.species_names
    )
    with pytest.raises(ValueError, match="requires a binary"):
        EquilibriumModel.from_state(ordinary, phase_equilibrium="spinodal")
    with pytest.raises((ValueError, BeartypeCallHintParamViolation)):
        state.with_phase_equilibrium(cast(PhaseEquilibriumMode, "typo"))


@pytest.mark.parametrize("problem", ["unknown", "gas", "background", "components"])
def test_configuration_errors_are_explicit(problem):
    kwargs: dict[str, Any] = dict(pressure=1.0, mass=0.0, metal_species=components())
    phase = "metal"
    if problem == "unknown":
        phase = "typo"
    elif problem == "gas":
        phase = "gas"
    elif problem == "background":
        kwargs["background_metal_mass"] = 1.0
    elif problem == "components":
        kwargs["metal_species"] = tuple(
            ChemicalSpecies.create_condensed(s, state="l") for s in ("Fe", "Si")
        )
    with pytest.raises(ValueError):
        ThermodynamicState.from_species(
            gases(), mixing_models={phase: SubregularSolution.regular_solution(50000)}, **kwargs
        )


def test_replacing_mixing_model_preserves_selected_mode():
    state = make_model("homogeneous").parameters.state
    replaced = state.with_mixing_models({"metal": SubregularSolution.regular_solution(0)})
    assert not metal(replaced).equilibrate
    assert metal(replaced).mixing_model.interaction_0 == 0
    assert metal(state).mixing_model.interaction_0 > 0


def test_closed_hydrogen_silicate_split_and_recombination():
    # A gas-reference linear term is harmless here: no gas or reaction exchanges
    # H2 with another species. Its choice cannot alter closed-binary coexistence.
    mixture = (
        ChemicalSpecies.create_solution("MgSiO3"),
        ChemicalSpecies.create_solution("H2", thermo=default_properties["H2_g"]),
    )
    model = EquilibriumModel.from_species(
        [],
        temperature=jnp.array([3500.0, 4400.0]),
        pressure=10000.0,
        mass=0.0,
        silicate_melt_species=mixture,
        mixing_models={"silicate_melt": SubregularSolution.hydrogen_silicate_rogers25()},
        phase_equilibrium="spinodal",
        mass_constraints={"H2": 0.01, "MgSiO3": 0.99},
    )
    for mode in ("homogeneous", "spinodal"):
        output = model.with_phase_equilibrium(mode).solve_with_default()
        assert np.all(output.multi_attempt_solution.success)
        data = output.to_dict(to_numpy=True)["silicate_melt"]
        lower, upper = [data["phase"]["subphases"][s] for s in ("lower", "upper")]
        for component in mixture:
            total = (
                lower["species_number_moles"][component.name]
                + upper["species_number_moles"][component.name]
            )
            np.testing.assert_allclose(
                total, data["species"]["number_moles"][component.name], rtol=1e-12
            )
        np.testing.assert_allclose(
            lower["mass"] + upper["mass"], data["phase"]["mass"], rtol=1e-12
        )
        np.testing.assert_array_equal(upper["present"].ravel(), [mode == "spinodal", False])
        mixing = data["phase"]["mixing"]
        delta = (
            mixing["homogeneous_mixing_gibbs_J_per_mol"] - mixing["actual_mixing_gibbs_J_per_mol"]
        )
        assert delta.min() > -1e-8
        if mode == "spinodal":
            assert delta[0, 0] > 100
            # This composition demixes despite positive curvature: nucleation
            # would be needed for the homogeneous phase to reach this equilibrium.
            assert mixing["metastable"][0, 0]


def test_dilute_subphase_inventory_is_not_lost_to_subtraction():
    phase = BinarySolutionPhase(components(), SubregularSolution.regular_solution(100 * R * 2000))
    output = phase.output(
        jnp.log(jnp.array([0.5, 0.5])),
        jnp.array([-30.0, -30.0]),
        jnp.array(2000.0),
        jnp.array(1.0),
    )
    assert isinstance(output, BinarySolutionPhaseOutput)
    assert output.subphases["upper"]["species_number_moles"]["Fe_l"].item() > 0
    np.testing.assert_allclose(
        output.subphases["upper"]["species_number_moles"]["Fe_l"], 0.5 * np.exp(-100), rtol=1e-10
    )
