# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thermodynamic databases in :mod:`atmodeller.database`"""

import importlib.resources
from pathlib import Path
from typing import Self

import equinox as eqx
import pytest

from atmodeller import override
from atmodeller.containers import ChemicalSpecies
from atmodeller.database import DataBase, GlennDataBase, get_default_database
from atmodeller.thermodata.core import DATA_DIRECTORY, ThermodynamicProperties
from atmodeller.thermodata.gustafson import diamond_gustafson
from atmodeller.thermodata.janaf import THERMODYNAMIC_DATA_SOURCE, glenn_properties


def test_default_database() -> None:
    """The default database gives the same species as creating them directly"""
    db: DataBase = get_default_database()

    assert eqx.tree_equal(db.create_gas("H2O"), ChemicalSpecies.create_gas("H2O"))
    for formula, state in (("H2O", "l"), ("C", "s"), ("C", "diamond")):
        assert eqx.tree_equal(
            db.create_condensed(formula, state=state),
            ChemicalSpecies.create_condensed(formula, state=state),
        )


def test_from_file(tmp_path: Path) -> None:
    """A database from a file matches the packaged data, without the species removed from it"""
    packaged = DATA_DIRECTORY.joinpath(THERMODYNAMIC_DATA_SOURCE)
    with importlib.resources.as_file(packaged) as path:  # type: ignore
        lines: list[str] = Path(path).read_text().splitlines(keepends=True)
    glenn_file: Path = tmp_path / "glenn.txt"
    glenn_file.write_text("".join(line for line in lines if not line.startswith("H2O,g,")))
    db: DataBase = GlennDataBase.from_file(glenn_file)

    assert eqx.tree_equal(db.create_gas("CO2"), GlennDataBase().create_gas("CO2"))
    with pytest.raises(KeyError, match="H2O_g not available"):
        db.create_gas("H2O")


def test_add_species() -> None:
    """Species are named from the Hill formula and replace existing ones, in this database only"""
    db: GlennDataBase = GlennDataBase()
    db.add_species("SiO2", "test", diamond_gustafson)
    db.add_species("C", "s", diamond_gustafson)

    assert db.create_condensed("SiO2", state="test").name == "O2Si_test"
    assert eqx.tree_equal(db.get_thermodynamic_properties("C_s"), diamond_gustafson)
    assert db.available_species().count("C_s") == 1
    assert "O2Si_test" not in glenn_properties
    assert "O2Si_test" not in GlennDataBase().available_species()


def test_custom_database() -> None:
    """A custom database only needs to provide its data"""

    class MyDataBase(DataBase):
        def __init__(self, coefficients: dict[str, ThermodynamicProperties]):
            self.coefficients = coefficients

        @override
        @classmethod
        def default(cls) -> Self:
            return cls({"H2O_g": glenn_properties["H2O_g"]})

        @override
        def set_thermodynamic_properties(
            self, name: str, properties: ThermodynamicProperties
        ) -> None:
            self.coefficients[name] = properties

        @override
        def available_species(self) -> tuple[str, ...]:
            return tuple(self.coefficients)

        @override
        def get_thermodynamic_properties(self, name: str) -> ThermodynamicProperties:
            return self.coefficients[name]

    db: DataBase = MyDataBase.default()

    assert eqx.tree_equal(db.create_gas("H2O"), ChemicalSpecies.create_gas("H2O"))
    with pytest.raises(KeyError, match=r"CO2_g not available. Available species are \('H2O_g',\)"):
        db.create_gas("CO2")
