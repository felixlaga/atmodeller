# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the integrated thermodynamic properties, the diamond of Gustafson (1986), and the
NASA Glenn data outside their temperature ranges"""

from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from scipy.optimize import brentq

from atmodeller.constants import TEMPERATURE_REFERENCE
from atmodeller.thermodata import (
    IntegratedThermodynamicProperties,
    ThermodynamicProperties,
)
from atmodeller.thermodata.gustafson import GRAPHITE_VOLUME_GUSTAFSON, diamond_gustafson
from atmodeller.thermodata.janaf import NasaGlennThermodynamicProperties, glenn_properties

graphite: NasaGlennThermodynamicProperties = cast(
    NasaGlennThermodynamicProperties, glenn_properties["C_s"]
)
"""Graphite from the NASA Glenn coefficients"""

TEMPERATURE: Array = jnp.linspace(200.0, 6000.0, 2901)
"""Temperatures in K spanning the NASA Glenn data for graphite"""


class GraphiteHeatCapacity(eqx.Module):
    """Heat capacity of graphite from the NASA Glenn coefficients, as a heat capacity model"""

    def cp(self, temperature):
        return graphite.cp(temperature)

    def temperature_range(self) -> tuple[float, float]:
        return graphite.temperature_range()


def test_integrated_matches_analytical() -> None:
    """Integrating the NASA Glenn heat capacity of graphite recovers its analytical H and S"""
    integrated: ThermodynamicProperties = IntegratedThermodynamicProperties(
        GraphiteHeatCapacity(),
        float(graphite.enthalpy(TEMPERATURE_REFERENCE)),
        float(graphite.entropy(TEMPERATURE_REFERENCE)),
    )

    np.testing.assert_allclose(
        integrated.enthalpy(TEMPERATURE), graphite.enthalpy(TEMPERATURE), rtol=0, atol=0.05
    )
    np.testing.assert_allclose(
        integrated.entropy(TEMPERATURE), graphite.entropy(TEMPERATURE), rtol=0, atol=1e-4
    )


def test_gustafson_gibbs_energy_exact() -> None:
    """Diamond of Gustafson is NASA Glenn graphite plus the Gibbs energy of diamond relative to
    graphite printed by Dinsdale (1991, p. 338), without quadrature error"""
    temperature: Array = jnp.linspace(TEMPERATURE_REFERENCE, 6000.0, 50)
    relative_printed: Array = (
        1009
        + 4.88 * temperature
        - 0.01 * temperature * jnp.log(temperature)
        + 135400 / temperature
        + 33.0e5 / temperature**2
        - 9e8 / temperature**3
    )
    np.testing.assert_allclose(
        diamond_gustafson.gibbs_energy(temperature) - graphite.gibbs_energy(temperature),
        relative_printed,
        atol=1e-6,
    )


def test_gustafson_reproduces_sgte_boundary() -> None:
    """Diamond and graphite with the SGTE volumes reproduce the transition of Gustafson (1986)"""
    graphite_gustafson: ThermodynamicProperties = graphite.with_volume(GRAPHITE_VOLUME_GUSTAFSON)

    def gibbs_difference(pressure: float, temperature: float) -> float:
        return float(
            diamond_gustafson.get_gibbs_over_RT(temperature, pressure)
            - graphite_gustafson.get_gibbs_over_RT(temperature, pressure)
        )

    # In K and GPa, from the SGTE Gibbs energies of graphite and diamond (Dinsdale 1991). At
    # 4710 K this is the graphite-diamond-liquid triple point at 12.4 GPa (Gustafson 1986)
    for temperature, pressure in (
        (298.15, 1.634),
        (1000.0, 3.360),
        (2000.0, 5.984),
        (3000.0, 8.502),
        (4000.0, 10.865),
        (4710.0, 12.447),
        (6000.0, 15.135),
    ):
        boundary: float = cast(float, brentq(gibbs_difference, 1.0, 1e6, args=(temperature,)))
        np.testing.assert_allclose(boundary / 1e4, pressure, atol=0.005)


def test_glenn_extrapolation_uses_nearest_range() -> None:
    """Outside its ranges, the NASA Glenn data use the nearest range, so G is continuous"""
    temperature: Array = jnp.array([199.0, 200.0, 6000.0, 6001.0, 7000.0])

    np.testing.assert_array_equal(graphite.get_index(temperature), [0, 0, 2, 2, 2])
    np.testing.assert_allclose(
        graphite.get_gibbs_over_RT(6000.0 + 1e-6, 1.0),
        graphite.get_gibbs_over_RT(6000.0, 1.0),
        rtol=1e-8,
    )
