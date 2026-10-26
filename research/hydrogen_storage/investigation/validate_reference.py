"""Numerical regression against the authors' immutable OSF workbooks."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from atmodeller.engine import compute_implied_log_stability
from model import make_meb, extract, serializable

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


def archive_guess(model, book, rows):
    columns = []
    for name in model.parameters.species_names:
        if name.endswith("_d"):
            s = name[:-2] + "_g"
            column = "dissolved_moles"
        elif name.endswith("_g"):
            s = name
            column = "atmosphere_moles"
        else:
            s = name
            column = "total_moles"
        columns.append(np.log(np.asarray(book[s][column].iloc[rows], dtype=float)))
    logn = np.column_stack(columns)
    stability = np.asarray(compute_implied_log_stability(model.parameters, logn))
    return np.concatenate([logn, stability], axis=1)


def main():
    start = time.time()
    h = np.tile(np.logspace(-1, 0, 50), 2)
    z = np.repeat([1, 100], 50)
    model, mass = make_meb(h, z)
    books = [pd.read_excel(HERE / "sources" / f"HHeCNOSi_magma_sol_real_{metal}xsolar_melt100.xlsx", sheet_name=None) for metal in [1,100]]
    # Stability is recomputed after concatenating, because constraints have batch size 100.
    columns = []
    for name in model.parameters.species_names:
        if name.endswith("_d"):
            s, column = name[:-2] + "_g", "dissolved_moles"
        elif name.endswith("_g"):
            s, column = name, "atmosphere_moles"
        else:
            s, column = name, "total_moles"
        columns.append(np.log(np.concatenate([np.asarray(b[s][column], float) for b in books])))
    logn = np.column_stack(columns)
    guess = np.concatenate([logn, np.asarray(compute_implied_log_stability(model.parameters, logn))], axis=1)
    print("Starting 100 archived MEB cases", flush=True)
    output = model.solve(guess)
    np.save(OUT / "reference_meb_solution.npy", np.asarray(output.solution))
    data = extract(output, mass)
    (OUT / "reference_meb.json").write_text(json.dumps(serializable(data), indent=2))
    comparison = {}
    for key, column in [("mole_fraction", "volume_mixing_ratio"), ("partial_pressure_bar", "pressure"), ("fugacity_bar", "fugacity")]:
        comparison[key] = {}
        for species, values in data[key].items():
            reference = np.concatenate([np.asarray(b[species][column], float) for b in books])
            comparison[key][species] = float(np.max(np.abs(values / reference - 1)))
    reference_pressure = np.concatenate([b['atmosphere'].pressure.to_numpy(float) for b in books])
    checks = {
        "n_cases": len(h), "n_success": int(sum(data["success"])),
        "maximum_relative_element_error": max(float(np.max(abs(v))) for v in data["relative_element_errors"].values()),
        "maximum_relative_pressure_difference": float(np.max(abs(data['pressure_bar']/reference_pressure-1))),
        "maximum_relative_species_differences": comparison,
        "seconds": time.time() - start,
    }
    (OUT / "reference_meb_validation.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
    print("Solar 1 wt%:", {s: v[49] for s,v in data['mole_fraction'].items()}, flush=True)


if __name__ == "__main__":
    main()
