# SPDX-License-Identifier: Apache-2.0

"""Matched homogeneous/demixed H2-MgSiO3 study through the public solver API.

The deep calculation is a closed binary. Atmospheric calculations are a separate,
one-way hot-envelope chemistry experiment with explicitly selected inventories.
No cooling path, kinetic transport law or additional partition coefficient is fitted.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import jax
from molmass import Formula
import numpy as np

from atmodeller import (
    ChemicalSpecies,
    EquilibriumModel,
    PurePhase,
    SolverParameters,
    SubregularSolution,
)
from atmodeller.engine import get_log_activity
from atmodeller.thermodata.defaults import default_properties

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GASES = ("H", "OH", "H2", "H2O", "SiH4", "Mg", "MgO", "O", "SiO", "O2", "SiO2", "O3", "Si")
CONDENSATES = (("MgO", "s"), ("SiO2", "l"), ("MgSiO3", "l"), ("Mg2SiO4", "l"), ("Si", "l"))
MOLAR_MASS = {f: Formula(f).mass * 1e-3 for f in ("H2", "MgSiO3")}


def portable(value):
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return portable(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path.write_text(json.dumps(portable(value), indent=2, allow_nan=False) + "\n")


def checks(output, data):
    n = output.parameters.batch_size
    names = [p.name for p in output.parameters.state.phase_system.phases if not p.is_empty]
    maximum = np.zeros(n)
    for element, target in output.parameters.mass_constraints.abundance_dict.items():
        target = np.broadcast_to(np.asarray(target).ravel(), (n,))
        total = sum(
            np.asarray(data[p]["elements"]["number_moles"].get(element, 0)).ravel() for p in names
        )
        if np.isfinite(target).all() and (target > 0).all():
            maximum = np.maximum(maximum, np.abs(total / target - 1))
    reaction = output.parameters.reaction_system.reaction
    log_n, log_stability = np.split(np.asarray(output.solution), 2, axis=-1)
    active = np.asarray(reaction.species.active_stability)
    stability = np.exp(np.where(active, log_stability, -np.inf))
    # Absent condensates can be undersaturated: their stability multiplier
    # belongs in the chemical-potential equations. Check complementarity
    # separately, so a large multiplier on appreciable material is rejected.
    complementarity = np.max(np.exp(log_n) * stability, axis=-1) / np.exp(log_n).sum(axis=-1)
    if reaction.number_reactions:
        log_activity = np.asarray(get_log_activity(output.parameters, output.solution))
        log_k = np.asarray(
            reaction.get_log_Kp(
                output.parameters.state.temperature, output.parameters.state.pressure
            )
        )
        residual = (log_activity - stability) @ reaction.get_matrix().T - log_k
        reaction_error = np.max(np.abs(residual), axis=-1)
    else:
        reaction_error = np.zeros(n)
    success = np.asarray(output.multi_attempt_solution.success)
    accepted = success & (maximum < 1e-8) & (reaction_error < 1e-7) & (complementarity < 1e-8)
    return {
        "success": success,
        "accepted": accepted,
        "maximum_relative_element_error": maximum,
        "maximum_abs_log_reaction_residual": reaction_error,
        "maximum_normalized_complementarity": complementarity,
    }


def deep_grid(quick):
    temperatures = [3000.0, 3500.0, 4000.0, 4250.0] if quick else np.arange(2500.0, 4401.0, 50.0)
    pressures = [10000.0] if quick else [10000.0, 30000.0, 50000.0, 100000.0]
    fractions = [1.0, 3.0] if quick else [0.1, 0.3, 1.0, 3.0, 5.0, 10.0]
    # Do not extend the phenomenological fit into its sign-reversed high-T
    # excess-energy regime. This is a chosen study window, not a calibration claim.
    return [
        {"id": i, "temperature_K": t, "pressure_bar": p, "hydrogen_wt_percent": w}
        for i, (p, t, w) in enumerate(
            (p, t, w)
            for p, t, w in itertools.product(pressures, temperatures, fractions)
            if 1 - t / 4670 + p / -350000 > 0
        )
    ]


def run_deep(directory, quick):
    rows = deep_grid(quick)
    t = np.array([r["temperature_K"] for r in rows])
    p = np.array([r["pressure_bar"] for r in rows])
    w = np.array([r["hydrogen_wt_percent"] for r in rows]) / 100
    components = (
        ChemicalSpecies.create_solution("MgSiO3"),
        ChemicalSpecies.create_solution("H2", state="fluid", thermo=default_properties["H2_g"]),
    )
    # H2's standard-state term is linear in component amounts. With no gas or
    # chemical conversion here, it cannot change closed-binary coexistence. It
    # is not being used as a calibrated high-pressure fluid chemical potential
    # for exchange with the atmospheric gas network.
    model = EquilibriumModel.from_species(
        [],
        temperature=t,
        pressure=p,
        mass=0.0,
        silicate_melt_species=components,
        mixing_models={"silicate_melt": SubregularSolution.hydrogen_silicate_rogers25()},
        phase_equilibrium="homogeneous",
        mass_constraints={"H2": w, "MgSiO3": 1 - w},
        solver_parameters=SolverParameters(atol=1e-10, rtol=1e-10, max_steps=256),
    )
    result = {
        "scope": "Closed H2-MgSiO3 binary, one kg total; no reactive atmospheric exchange.",
        "grid": rows,
        "molar_masses_kg_per_mol": MOLAR_MASS,
        "modes": {},
    }
    for mode in ("homogeneous", "spinodal"):
        started = time.perf_counter()
        output = model.with_phase_equilibrium(mode).solve_with_default()
        data = output.to_dict(to_numpy=True)
        validation = checks(output, data)
        phase = data["silicate_melt"]["phase"]
        validation["accepted"] &= phase["mixing"]["valid"].ravel()
        result["modes"][mode] = {
            "validation": validation,
            "phase": phase,
            "species": data["silicate_melt"]["species"],
            "seconds": time.perf_counter() - started,
        }
        np.save(directory / f"deep_{mode}_solution.npy", np.asarray(output.solution))
        print(f"deep {mode}: {validation['accepted'].sum()}/{len(rows)} accepted", flush=True)
    save(directory / "deep.json", result)
    return portable(result)


def feeds_from_deep(deep, quick) -> list[dict[str, Any]]:
    temperatures = (
        [3000.0, 3500.0, 4000.0, 4250.0]
        if quick
        else [2500.0, 3000.0, 3500.0, 3750.0, 4000.0, 4250.0]
    )
    fractions = [0.0, 1.0] if quick else [0.0, 0.01, 0.1, 1.0]
    rows: list[dict[str, Any]] = []
    for parent in deep["grid"]:
        if parent["pressure_bar"] != 10000.0 or parent["temperature_K"] not in temperatures:
            continue
        i = parent["id"]
        for mode in ("homogeneous", "spinodal"):
            values = deep["modes"][mode]
            if not values["validation"]["accepted"][i]:
                continue
            phase = values["phase"]
            lower, upper = [phase["subphases"][s] for s in ("lower", "upper")]
            separated = bool(upper["present"][i][0])
            for eta in [1.0] if mode == "homogeneous" else fractions:
                moles = {}
                for formula, name in (("MgSiO3", "MgO3Si_l"), ("H2", "H2_fluid")):
                    lower_n = lower["species_number_moles"][name][i][0]
                    upper_n = upper["species_number_moles"][name][i][0]
                    moles[formula] = upper_n + eta * lower_n if separated else lower_n + upper_n
                masses = {f: moles[f] * MOLAR_MASS[f] for f in moles}
                available_mass = sum(masses.values())
                rows.append(
                    {
                        "feed_id": len(rows),
                        "parent_id": i,
                        "mode": mode,
                        "entrained_lower_fraction": eta,
                        "deep_temperature_K": parent["temperature_K"],
                        "deep_pressure_bar": parent["pressure_bar"],
                        "hydrogen_wt_percent": parent["hydrogen_wt_percent"],
                        "separated": separated,
                        "available_mass_kg_per_kg_bulk": available_mass,
                        "available_component_moles": moles,
                        "normalized_component_mass_kg": {
                            f: m / available_mass for f, m in masses.items()
                        },
                        "Si_to_H_atomic_ratio": moles["MgSiO3"] / (2 * moles["H2"]),
                    }
                )
    return rows


def run_atmospheres(directory, deep, quick):
    feeds = feeds_from_deep(deep, quick)
    temperatures = [2500.0] if quick else [2300.0, 2500.0, 2800.0, 3000.0]
    pressures = [0.1, 1.0] if quick else [0.01, 0.1, 1.0, 10.0]
    rows = [
        dict(feed, atmosphere_temperature_K=t, atmosphere_pressure_bar=p)
        for feed, t, p in itertools.product(feeds, temperatures, pressures)
    ]
    t = np.array([r["atmosphere_temperature_K"] for r in rows])
    p = np.array([r["atmosphere_pressure_bar"] for r in rows])
    masses = {
        f: np.array([r["normalized_component_mass_kg"][f] for r in rows]) for f in MOLAR_MASS
    }
    gas = tuple(ChemicalSpecies.create_gas(s) for s in GASES)
    solids = tuple(PurePhase.from_species(s, state=phase) for s, phase in CONDENSATES)
    for component in (*gas, *(phase.species[0] for phase in solids)):
        assert isinstance(component, ChemicalSpecies)
        lo, hi = component.thermo.temperature_range()
        if t.min() < lo or t.max() > hi:
            raise ValueError(f"Temperature outside data range of {component.name}.")
    save(directory / "feeds.json", feeds)
    for condensed in (True, False):
        label = "condensed" if condensed else "gas_only"
        started = time.perf_counter()
        model = EquilibriumModel.from_species(
            gas,
            temperature=t,
            pressure=p,
            mass=0.0,
            condensates=solids if condensed else (),
            mass_constraints=masses,
            solver_parameters=SolverParameters(atol=1e-10, rtol=1e-10, max_steps=256),
        )
        output = model.solve_with_default()
        data = output.to_dict(to_numpy=True)
        validation = checks(output, data)
        total_pressure = sum(
            np.asarray(v).ravel() for v in data["gas"]["species"]["partial_pressure"].values()
        )
        validation["relative_pressure_error"] = np.abs(total_pressure / p - 1)
        validation["accepted"] &= validation["relative_pressure_error"] < 1e-9
        result = {
            "scope": "One-way hot-envelope parcel equilibrium; conditional phase selection/entrainment, not a planetary cooling model.",
            "cases": rows,
            "validation": validation,
            "gas": data["gas"],
            "condensates": {
                key: val for key, val in data.items() if key in [s.name for s in solids]
            },
            "network": {"gas": GASES, "condensates": CONDENSATES if condensed else []},
            "seconds": time.perf_counter() - started,
        }
        save(directory / f"atmosphere_{label}.json", result)
        np.save(directory / f"atmosphere_{label}_solution.npy", np.asarray(output.solution))
        print(
            f"atmosphere {label}: {validation['accepted'].sum()}/{len(rows)} accepted; "
            f"max element error {validation['maximum_relative_element_error'].max():.3g}; "
            f"max lnQ/K {validation['maximum_abs_log_reaction_residual'].max():.3g}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stage", choices=("deep", "atmosphere", "all"), default="all")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(HERE / ".jax_cache"))
    if args.stage in ("deep", "all"):
        deep = run_deep(directory, args.quick)
    else:
        deep = json.loads((directory / "deep.json").read_text())
    if args.stage in ("atmosphere", "all"):
        run_atmospheres(directory, deep, args.quick)
    source_paths = [
        HERE / "run_study.py",
        *sorted((REPO / "atmodeller").rglob("*.py")),
        REPO / "uv.lock",
        REPO / "atmodeller/thermodata/data/nasa_glenn_coefficients.txt",
    ]
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "quick": args.quick,
        "stage": args.stage,
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(REPO), "status", "--porcelain"], text=True
            ).strip()
        ),
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir())
            if p.name
            in {
                "deep.json",
                "deep_homogeneous_solution.npy",
                "deep_spinodal_solution.npy",
                "feeds.json",
                "atmosphere_condensed.json",
                "atmosphere_condensed_solution.npy",
                "atmosphere_gas_only.json",
                "atmosphere_gas_only_solution.npy",
            }
        },
        "literature": [
            "https://arxiv.org/html/2509.13320v1",
            "https://doi.org/10.1093/mnras/staf1940",
            "https://doi.org/10.1093/mnras/stag133",
        ],
        "hypotheses": [
            "Binary formula-unit free energy; no other deep components.",
            "The separated upper phase can supply an atmospheric parcel.",
            "The lower-phase entrainment fraction is an imposed control, not a computed transport law.",
            "Atmospheric equilibrium is one-way and does not feed back on deep demixing.",
            "Ideal gas at 0.01-10 bar; hot 2300-3000 K condensates only; no cold-cloud or spectral prediction.",
        ],
    }
    save(directory / "provenance.json", provenance)


if __name__ == "__main__":
    main()
