# SPDX-License-Identifier: Apache-2.0

"""Check recorded sources/results, optionally re-solve perturbed atmospheric cases."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from atmodeller import ChemicalSpecies, EquilibriumModel, PurePhase, SolverParameters
from run_study import CONDENSATES, GASES, MOLAR_MASS, checks, save

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "outputs"


def verify_files():
    provenance = json.loads((OUT / "provenance.json").read_text())
    for root, key in ((REPO, "source_sha256"), (OUT, "files_sha256")):
        for name, expected in provenance[key].items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            assert actual == expected, f"Changed since study: {name}"
    for name in ("deep", "atmosphere_condensed", "atmosphere_gas_only"):
        data = json.loads((OUT / f"{name}.json").read_text())
        values = data["modes"].values() if name == "deep" else [data]
        for value in values:
            assert all(value["validation"]["accepted"]), f"Unaccepted result: {name}"
    print("All recorded source/data hashes match; every stored solve is accepted.")


def resolve():
    jax.config.update("jax_compilation_cache_dir", str(HERE / ".jax_cache"))
    result = {"relative_log_tolerance": 1e-8, "atol": 1e-12, "rtol": 1e-12, "networks": {}}
    for label in ("condensed", "gas_only"):
        data = json.loads((OUT / f"atmosphere_{label}.json").read_text())
        chosen = [
            i
            for i, row in enumerate(data["cases"])
            if row["deep_temperature_K"] == 3500
            and row["hydrogen_wt_percent"] in (1, 3)
            and row["atmosphere_temperature_K"] == 2500
            and row["atmosphere_pressure_bar"] == 1
        ]
        indices = np.unique([*np.linspace(0, len(data["cases"]) - 1, 24, dtype=int), *chosen])
        rows = [data["cases"][i] for i in indices]
        gas = tuple(ChemicalSpecies.create_gas(s) for s in GASES)
        condensed = tuple(PurePhase.from_species(s, state=phase) for s, phase in CONDENSATES)
        model = EquilibriumModel.from_species(
            gas,
            temperature=np.array([r["atmosphere_temperature_K"] for r in rows]),
            pressure=np.array([r["atmosphere_pressure_bar"] for r in rows]),
            mass=0.0,
            condensates=condensed if label == "condensed" else (),
            mass_constraints={
                f: np.array([r["normalized_component_mass_kg"][f] for r in rows])
                for f in MOLAR_MASS
            },
            solver_parameters=SolverParameters(atol=1e-12, rtol=1e-12, max_steps=256),
        )
        solution = np.load(OUT / f"atmosphere_{label}_solution.npy")[indices]
        # Perturb both amounts and stability variables, so the saved solution
        # is not itself accepted as an initial converged answer.
        perturbed = solution + 0.4 * np.sin(np.arange(solution.size).reshape(solution.shape) + 1)
        output = model.solve(jnp.asarray(perturbed))
        recomputed = output.to_dict(to_numpy=True)
        validation = checks(output, recomputed)
        assert validation["accepted"].all()
        gas_differences = {}
        for name, values in data["gas"]["species"]["mole_fraction"].items():
            old = np.asarray(values).ravel()[indices]
            new = np.asarray(recomputed["gas"]["species"]["mole_fraction"][name]).ravel()
            difference = float(np.max(np.abs(np.log(new / old))))
            assert difference < 1e-8, (label, name, difference)
            gas_differences[name] = difference
        result["networks"][label] = {
            "case_indices": indices,
            "validation": validation,
            "maximum_gas_abs_log_ratio_by_species": gas_differences,
        }
        print(f"{label}: {len(indices)} perturbed-start/tighter-tolerance re-solves agree.")
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(HERE / "numerical_crosscheck.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", action="store_true")
    args = parser.parse_args()
    verify_files()
    if args.resolve:
        resolve()
