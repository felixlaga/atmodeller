# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the EOS models from :cite:t:`CD21`"""

import pytest

from atmodeller.eos import RealGas
from atmodeller.eos._chabrier import (
    H2_3000K_chabrier21,
    H2_4000K_chabrier21,
    H2_chabrier21,
    H2_He_Y0275_chabrier21,
    H2_He_Y0292_chabrier21,
    H2_He_Y0297_chabrier21,
    He_chabrier21,
    _H2_3000K_chabrier21,
)
from atmodeller.eos.core import RealGasBase
from atmodeller.sci_utils import unit_conversion


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(H2_chabrier21, id="H"),
        pytest.param(He_chabrier21, id="He"),
        pytest.param(H2_He_Y0275_chabrier21, id="Y0275"),
        pytest.param(H2_He_Y0292_chabrier21, id="Y0292"),
        pytest.param(H2_He_Y0297_chabrier21, id="Y0297"),
    ],
)
def test_ideal_gas_limit(check_values, model: RealGas) -> None:
    """Tests that the compressibility factor tends to unity at low pressure

    For the H2-He mixtures this checks that Y is treated as a He mass fraction when converting the
    tabulated mass density to molar density. Treating Y as a mole fraction gives Z of about 1.1.
    """
    check_values.compressibility_factor(1000, 1, model, 1.0, rtol=2e-3, atol=0)


def test_H2_volume_100kbar(check_values) -> None:
    """Tests volume at 100 kbar"""
    expected: float = 9.005066169376918
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(3000, 100e3, H2_chabrier21, expected)


def test_H2_fugacity_coefficient_100kbar(check_values) -> None:
    """Tests fugacity coefficient at 100 kbar"""
    # Assumes Simpson integration with 101 nodes
    expected: float = 32.808953401650605
    check_values.fugacity_coefficient(3000, 100e3, H2_chabrier21, expected)


def test_H2_volume_1000kbar(check_values) -> None:
    """Tests volume at 1000 kbar"""
    expected: float = 3.0100820540769166
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(5000, 1000e3, H2_chabrier21, expected)


def test_H2_fugacity_coefficient_1000kbar(check_values) -> None:
    """Tests fugacity coefficient at 1000 kbar"""
    # Assumes Simpson integration with 101 nodes
    expected: float = 450833.8733449097
    check_values.fugacity_coefficient(5000, 1000e3, H2_chabrier21, expected)


def test_broadcasting(check_values) -> None:
    """Tests methods with broadcasting"""
    model: RealGas = H2_chabrier21
    check_values.check_broadcasting(model)


def test_H2_3000K_ideal_branch(check_values) -> None:
    """Tests the ideal gas (extrapolated) branch below 1 bar for the 3000 K function fit

    This exercises :meth:`CombinedRealGasFugacity._get_index` and
    :meth:`CombinedRealGasFugacity.log_fugacity` for the first (ideal gas) EOS in the composite.
    """
    model: RealGasBase = H2_3000K_chabrier21
    # An ideal gas has a fugacity coefficient of unity by construction
    expected: float = 1.0
    check_values.fugacity_coefficient(3000, 0.5, model, expected)


def test_H2_3000K_fit_branch(check_values) -> None:
    """Tests the Chabrier function-fit branch above 1 bar for the 3000 K function fit"""
    model: RealGasBase = H2_3000K_chabrier21
    expected: float = 1.188884415391348
    check_values.fugacity_coefficient(3000, 100, model, expected)
    expected_log_fugacity: float = 4.778185587357713
    check_values._check_property("log_fugacity", 3000, 100, model, expected_log_fugacity)


def test_H2_4000K_ideal_branch(check_values) -> None:
    """Tests the ideal gas (extrapolated) branch below 1 bar for the 4000 K function fit"""
    model: RealGasBase = H2_4000K_chabrier21
    expected: float = 1.0
    check_values.fugacity_coefficient(4000, 0.5, model, expected)


def test_H2_4000K_fit_branch(check_values) -> None:
    """Tests the Chabrier function-fit branch above 1 bar for the 4000 K function fit"""
    model: RealGasBase = H2_4000K_chabrier21
    expected: float = 4.206900056983392
    check_values.fugacity_coefficient(4000, 1000, model, expected)
    expected_log_fugacity: float = 8.344481326945935
    check_values._check_property("log_fugacity", 4000, 1000, model, expected_log_fugacity)


def test_ChabrierFunction_raw_log_fugacity(check_values) -> None:
    """Tests the raw ChabrierFunction class's log_fugacity override directly

    :class:`~atmodeller.eos._aggregators.CombinedRealGasFugacity` only ever calls
    ``log_fugacity_coefficient`` on its constituent EOS models (not ``log_fugacity``), so
    :meth:`~atmodeller.eos._chabrier.ChabrierFunction.log_fugacity` is otherwise never exercised.
    """
    model: RealGasBase = _H2_3000K_chabrier21
    expected: float = 4.778185587357713
    check_values._check_property("log_fugacity", 3000, 100, model, expected)
