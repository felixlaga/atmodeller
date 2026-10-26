"""Run the upstream README H-O example and check tracked-element conservation.

Run from this workspace: .venv/bin/python run_baseline.py
This is an installation/solver check, not a reproduction of Hakim (2026),
and includes no spinodal or phase-separation extension.
"""
import json
from pathlib import Path
import subprocess

import jax
import numpy as np
import atmodeller
from atmodeller import (
    ChemicalSpecies, EquilibriumModel, Parameters, Planet,
    ReservoirSpecies, earth, get_solubility_models,
)

ROOT = Path(__file__).resolve().parent
gas = tuple(ChemicalSpecies.create_gas(s) for s in ("H2", "H2O", "O2"))
water = ReservoirSpecies.create_dissolved(
    "H2O", solubility=get_solubility_models()["H2O_peridotite_sossi23"]
)
planet = Planet.from_species(gas, silicate_melt_species=(water,))
budgets = {"H": float(earth.oceans_to_hydrogen_mass(1)), "O": 6.25774e20}
output = EquilibriumModel(Parameters(planet, mass_constraints=budgets)).solve_with_default()
assert np.all(np.asarray(output.multi_attempt_solution.success)), "Solver failed"
data = output.to_dict(to_numpy=True)

errors = {}
for element, target in budgets.items():
    total = sum(
        np.asarray(data[phase.name].get("elements", {}).get("mass", {}).get(element, 0.0))
        for phase in planet.phase_system.phases
        if not phase.is_empty
    )
    error = float(np.max(np.abs(total - target) / target))
    assert error < 1e-6, (element, error)
    errors[element] = error

pressures = {
    species: float(np.asarray(value).item())
    for species, value in data["gas"]["species"]["partial_pressure"].items()
}
assert all(np.isfinite(v) and v > 0 for v in pressures.values())
result = {
    "scope": "Upstream README H-O example; no Hakim reproduction or spinodal physics",
    "atmodeller_version": atmodeller.__version__,
    "commit": subprocess.check_output(["git", "-C", str(ROOT.parents[1]), "rev-parse", "HEAD"], text=True).strip(),
    "jax_version": jax.__version__,
    "devices": [str(d) for d in jax.devices()],
    "solver_success": True,
    "temperature_K": float(np.asarray(planet.temperature).item()),
    "element_budgets_kg": budgets,
    "element_relative_mass_errors": errors,
    "gas_partial_pressures_bar": pressures,
}
directory = ROOT / "outputs"
directory.mkdir(exist_ok=True)
(directory / "baseline_check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
