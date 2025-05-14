#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Tests for systems with real gases"""

import logging
from typing import Mapping

import numpy as np
import pytest
from jax.typing import ArrayLike

from atmodeller import debug_logger
from atmodeller.classes import InteriorAtmosphere, SolverParameters
from atmodeller.containers import (
    ConstantFugacityConstraint,
    Planet,
    Species,
    SpeciesCollection,
)
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import (
    ActivityProtocol,
    FugacityConstraintProtocol,
    SolubilityProtocol,
)
from atmodeller.output import Output
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()
eos_models: Mapping[str, ActivityProtocol] = get_eos_models()


def test_fO2_holley(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`HWZ58`"""

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_beattie_holley58"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")

    species: SpeciesCollection = SpeciesCollection((H2_g, H2O_g, O2_g))
    # Temperature is within the range of the Holley model
    planet: Planet = Planet(surface_temperature=1000)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {
        "H": h_kg,
    }

    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 32.77037875523393,
        "H2_g": 71.50338102110962,
        "O2_g": 1.525466019972294e-21,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_earth(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21`"""

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")
    SiO_g: Species = Species.create_gas("OSi_g")
    H4Si_g: Species = Species.create_gas("H4Si_g")
    O2Si_l: Species = Species.create_condensed("O2Si_l")

    species: SpeciesCollection = SpeciesCollection((H2_g, H2O_g, O2_g, H4Si_g, SiO_g, O2Si_l))
    planet: Planet = Planet(surface_temperature=3400)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = h_kg * 10
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    interior_atmosphere.solve(planet=planet, mass_constraints=mass_constraints)
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 7.253556287801738e03,
        "H2O_g_activity": 7.253556287801635e03,
        "H2_g": 1.162520652380062e04,
        "H2_g_activity": 2.516876841308367e05,
        "H4Si_g": 6.759146395057408e04,
        "H4Si_g_activity": 6.759146395057408e04,
        "O2Si_l": 9.311489514762553e04,
        "O2Si_l_activity": 1.0,
        "O2_g": 1.791815879185495e-05,
        "O2_g_activity": 1.791815879185482e-05,
        "OSi_g": 6.302402285027329e02,
        "OSi_g_activity": 6.302402285027240e02,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune

    This case effectively saturates the maximum allowable log number density at a value of 70
    based on the default hypercube that brackets the solution (see LOG_NUMBER_DENSITY_UPPER).
    This is fine for a test, but this test is not physically realistic because solubilities are
    ignored, which would greatly lower the pressure and hence the number density.
    """

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")
    OSi_g: Species = Species.create_gas("OSi_g")
    H4Si_g: Species = Species.create_gas("H4Si_g")
    O2Si_l: Species = Species.create_condensed("O2Si_l")

    species: SpeciesCollection = SpeciesCollection((H2_g, H2O_g, O2_g, H4Si_g, OSi_g, O2Si_l))

    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet(
        surface_temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = 6.74717e24

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    interior_atmosphere.solve(
        planet=planet,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 4.295071823974879e05,
        "H2O_g_activity": 4.295071823974879e05,
        "H2_g": 2.926773356736283e00,
        "H2_g_activity": 1.956449985411128e04,
        "H4Si_g": 7.038499826508187e-04,
        "H4Si_g_activity": 7.038499826508187e-04,
        "O2Si_l": 4.497910721606553e05,
        "O2Si_l_activity": 1.0,
        "O2_g": 1.039725511931324e01,
        "O2_g_activity": 1.039725511931332e01,
        "OSi_g": 8.273579821046055e-01,
        "OSi_g_activity": 8.273579821046055e-01,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune_batch(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune for several O masses

    H2O solubility is also included.

    As above, this test has questionable physical relevance without the inclusion of more species'
    solubility, but it serves its purpose as a test.
    """
    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21"])
    H2O_g: Species = Species.create_gas(
        "H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    O2_g: Species = Species.create_gas("O2_g")
    OSi_g: Species = Species.create_gas("OSi_g")
    H4Si_g: Species = Species.create_gas("H4Si_g")
    O2Si_l: Species = Species.create_condensed("O2Si_l")

    species: SpeciesCollection = SpeciesCollection((H2_g, H2O_g, O2_g, H4Si_g, OSi_g, O2Si_l))

    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet(
        surface_temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    # Batch solve for three oxygen masses
    o_kg: ArrayLike = 1e24 * np.array([7.0, 7.5, 8.0])

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    interior_atmosphere.solve(planet=planet, mass_constraints=mass_constraints)
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array([3.415377850792279e04, 3.464713381927368e04, 3.477346411670301e04]),
        "H2_g": np.array([1.953887558935978e00, 1.649313472441684e-01, 2.675210310320718e-02]),
        "H2_g_activity": np.array(
            [2.695007407667841e01, 1.476984016623652e01, 1.127184619297398e01]
        ),
        "O2_g": np.array([3.464758174266921e04, 1.187130259426918e05, 2.053152202141989e05]),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_pH2_fO2_real_gas(helper) -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the fugacity of H2.
    """
    H2O_g: Species = Species.create_gas(
        "H2O_g",
        solubility=solubility_models["H2O_peridotite_sossi23"],
        activity=eos_models["H2O_cork_holland98"],
    )
    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_cork_cs_holland91"])
    O2_g: Species = Species.create_gas("O2_g")

    species: SpeciesCollection = SpeciesCollection((H2O_g, H2_g, O2_g))
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    # FIXME: Probably breaking because the order of the fugacity constraints is not
    # consistent with the order of the species in the collection
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": ConstantFugacityConstraint(1.0453574209588085e-07),
        # Gives a H2 partial pressure of around 1000 bar
        "H2_g": ConstantFugacityConstraint(1493.1),
    }

    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 1470.2567650857518,
        "H2_g": 999.9971214963639,
        "O2_g": 1.045357420958815e-07,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


@pytest.mark.skip(reason="Can fail if multistart is not large enough")
def test_H_and_C_real_gas(helper) -> None:
    """Tests H2-H2O-O2-CO-CO2-CH4 at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`."""

    H2_g: Species = Species.create_gas(
        "H2_g",
        solubility=solubility_models["H2_basalt_hirschmann12"],
        activity=eos_models["H2_cork_cs_holland91"],
    )
    H2O_g: Species = Species.create_gas(
        "H2O_g",
        solubility=solubility_models["H2O_peridotite_sossi23"],
        activity=eos_models["H2O_cork_holland98"],
    )
    O2_g: Species = Species.create_gas("O2_g")
    CO_g: Species = Species.create_gas("CO_g", activity=eos_models["CO_cork_cs_holland91"])
    CO2_g: Species = Species.create_gas(
        "CO2_g",
        solubility=solubility_models["CO2_basalt_dixon95"],
        activity=eos_models["CO2_cork_holland98"],
    )
    CH4_g: Species = Species.create_gas("CH4_g", activity=eos_models["CH4_cork_cs_holland91"])

    species: SpeciesCollection = SpeciesCollection((H2_g, H2O_g, O2_g, CO_g, CO2_g, CH4_g))
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "H2_g": ConstantFugacityConstraint(958),
        "O2_g": ConstantFugacityConstraint(1.0132255325169718e-07),
    }

    oceans: ArrayLike = 10
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg}

    solver_parameters = SolverParameters(multistart=5)

    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
        solver_parameters=solver_parameters,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "CH4_g": 1.161221651357397e01,
        "CO2_g": 6.719430723567530e01,
        "CO_g": 2.768796027177136e02,
        "H2O_g": 9.552659883631408e02,
        "H2_g": 6.942030982137493e02,
        "O2_g": 1.013225532516968e-07,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)
