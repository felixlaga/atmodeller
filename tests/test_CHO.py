# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for C-H-O systems

The comparisons with FactSage and FastChem are rows of a single batched model of C-H-O gas with
graphite, so that the solver is compiled once. Graphite is unstable in the gas-only comparisons.
"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from jaxtyping import ArrayLike

from atmodeller import debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.interfaces import ActivityConstraintProtocol, SolubilityProtocol, SpeciesProtocol
from atmodeller.output import Output
from atmodeller.output_base import OutputNamedArraysDict
from atmodeller.parameters import Parameters
from atmodeller.phases import PurePhase
from atmodeller.sci_utils import earth
from atmodeller.solubility import get_solubility_models
from atmodeller.state import BaseThermodynamicState, Planet
from atmodeller.thermodata import IronWustiteBuffer

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()

# Gas species
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO")
CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2")
CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4")

# Melt species
H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
    "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
)
CO2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
    "CO2", solubility=solubility_models["CO2_basalt_dixon95"], include_in_phase_mass=False
)

OCEAN: float = float(earth.oceans_to_hydrogen_mass(1))
"""Mass of H in one Earth ocean (kg)"""

# Each case is (temperature, fO2 shift relative to IW, H kg, C kg, O kg, references). A NaN fO2
# shift or O mass is unconstrained. References are FactSage, unless labelled FastChem.
CASES: dict[str, tuple[float, float, float, float, float, list[dict[str, Any]]]] = {
    # Similar to BHS22, Table E, row 1
    "IW-2": (
        1400, -2, 3 * OCEAN, 3 * OCEAN, np.nan,
        [{"partial_pressure": {"H2_g": 175.5, "H2O_g": 13.8, "CO_g": 6.21, "CO2_g": 0.228,
                               "CH4_g": 38.07, "O2_g": 1.25e-15}}],
    ),
    # Similar to BHS22, Table E, row 2. Graphite is unstable.
    "IW+0.5": (
        1400, 0.5, 3 * OCEAN, 3 * OCEAN, np.nan,
        [
            {"partial_pressure": {"CH4_g": 28.66, "CO2_g": 30.88, "CO_g": 46.42,
                                  "H2O_g": 337.16, "H2_g": 236.98, "O2_g": 4.11e-13},
             "C_s_activity": 0.12202},
            # FastChem
            {"partial_pressure": {"CH4_g": 29.61919788, "CO2_g": 29.82548282,
                                  "CO_g": 45.94958264, "H2O_g": 332.03616807,
                                  "H2_g": 236.73845646, "O2_g": 3.96475584e-13}},
        ],
    ),
    # Similar to BHS22, Table E, row 3
    "IW+2": (
        1400, 2, OCEAN, 0.1 * OCEAN, np.nan,
        [{"partial_pressure": {"CH4_g": 0.00129, "CO2_g": 3.25, "CO_g": 0.873, "H2O_g": 218.48,
                               "H2_g": 27.40, "O2_g": 1.29e-11}}],
    ),
    # Similar to BHS22, Table E, row 4
    "IW+4": (
        1400, 4, OCEAN, 5 * OCEAN, np.nan,
        [{"partial_pressure": {"CH4_g": 7.13e-05, "CO2_g": 357.23, "CO_g": 10.21,
                               "H2O_g": 432.08, "H2_g": 5.78, "O2_g": 1.14e-09}}],
    ),
    "873K": (
        873, 0, OCEAN, OCEAN, np.nan,
        [{"partial_pressure": {"H2_g": 59.066, "H2O_g": 18.320, "CO_g": 8.91e-4,
                               "CO2_g": 7.48e-4, "CH4_g": 19.548, "O2_g": 1.27e-25}}],
    ),
    "450K": (
        450, np.nan, OCEAN, OCEAN, 1.02999e20,
        [{"partial_pressure": {"H2_g": 55.475, "H2O_g": 8.0, "CO2_g": 1.24e-14,
                               "O2_g": 7.85e-54, "CH4_g": 16.037, "CO_g": 2.12e-16}}],
    ),
    # Graphite stable with around 50% condensed C mass fraction
    "graphite_stable": (
        873, np.nan, OCEAN, 5 * OCEAN, 2.73159e19,
        [{"partial_pressure": {"O2_g": 1.27e-25, "H2_g": 14.564, "CO_g": 0.07276,
                               "H2O_g": 4.527, "CO2_g": 0.061195, "CH4_g": 96.74},
          "C_s_activity": 1.0, "C_s_mass": 3.54162e20}],
    ),
}  # fmt: skip


@pytest.fixture(scope="module")
def CHO_output() -> dict[str, Any]:
    """Output of the C-H-O model with graphite, solved for all cases in one batch"""
    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)
    condensates: tuple[PurePhase, ...] = (PurePhase.from_species("C", state="s"),)

    def column(index: int) -> ArrayLike:
        return np.array([case[index] for case in CASES.values()], dtype=float)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, temperature=column(0), condensates=condensates
    )
    activity_constraints: dict[str, ActivityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(column(1))
    }
    mass_constraints: dict[str, ArrayLike] = {"H": column(2), "C": column(3), "O": column(4)}
    parameters: Parameters = Parameters(
        planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )

    output: Output = EquilibriumModel(parameters).solve_with_default()
    assert bool(np.all(output.multi_attempt_solution.solver_success))

    return OutputNamedArraysDict(
        output.parameters, output.multi_attempt_solution, output.initial_guess
    ).to_dict(expand_to_batch=True, ravel=True, to_numpy=True)


@pytest.mark.parametrize("case", CASES)
def test_CHO(CHO_output: dict[str, Any], case: str) -> None:
    """Partial pressures and graphite match FactSage and FastChem"""
    row: int = list(CASES).index(case)
    graphite: dict[str, Any] = CHO_output["C_s"]["species"]

    for reference in CASES[case][-1]:
        computed: dict[str, float] = {
            name: CHO_output["gas"]["species"]["partial_pressure"][name][row]
            for name in reference["partial_pressure"]
        }
        expected: dict[str, float] = dict(reference["partial_pressure"])
        if "C_s_activity" in reference:
            computed["C_s_activity"] = graphite["activity"]["C_s"][row]
            expected["C_s_activity"] = reference["C_s_activity"]
        if "C_s_mass" in reference:
            computed["C_s_mass"] = graphite["mass"]["C_s"][row]
            expected["C_s_mass"] = reference["C_s_mass"]

        np.testing.assert_allclose(
            np.log10(list(computed.values())),
            np.log10(list(expected.values())),
            rtol=TOLERANCE,
            atol=TOLERANCE,
            err_msg=f"{case}: {list(expected)}",
        )


def test_H_and_C() -> None:
    """Tests H2-H2O and CO-CO2 with H2O and CO2 solubility."""
    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g)
    melt_species: tuple[SpeciesProtocol, ...] = (H2O_d, CO2_d)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, silicate_melt_species=melt_species
    )
    activity_constraints: dict[str, ActivityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}
    mass_constraints: dict[str, ArrayLike] = {"C": OCEAN, "H": OCEAN}
    parameters: Parameters = Parameters(
        planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )

    output: Output = EquilibriumModel(parameters).solve_with_default()

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 0.2582458752325180,
                    "H2_g": 0.2502809714412906,
                    "O2_g": 8.838513516896038e-08,
                    "CO_g": 59.65835224848439,
                    "CO2_g": 13.43793686555727,
                }
            }
        }
    }

    assert output.compare(target, rtol=TOLERANCE, atol=TOLERANCE, log=False)
