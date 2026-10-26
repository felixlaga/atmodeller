# SPDX-License-Identifier: Apache-2.0

"""Analyze conserved inventories and matched controls; export research figures."""

import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from molmass import Formula
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize_scalar

from atmodeller import SubregularSolution
from atmodeller.sci_utils import GAS_CONSTANT as R
from run_study import MOLAR_MASS, save

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = HERE / "figures"
COLORS = ["#167c80", "#df8842", "#50548c", "#345069"]


def array(value):
    return np.asarray(value, dtype=float).ravel()


def gas_name(formula):
    return f"{Formula(formula).formula}_g"


def atmosphere_frame(data):
    frame = pd.DataFrame(data["cases"])
    for formula in ("H2", "H2O", "SiO", "SiH4", "Mg", "O2"):
        frame[formula] = array(data["gas"]["species"]["mole_fraction"][gas_name(formula)])
    frame["mean_molar_mass_g_per_mol"] = array(data["gas"]["phase"]["molar_mass"]) * 1000
    frame["gas_mass_kg_per_kg_feed"] = array(data["gas"]["phase"]["mass"])
    frame["gas_mass_kg_per_kg_bulk"] = (
        frame["gas_mass_kg_per_kg_feed"] * frame["available_mass_kg_per_kg_bulk"]
    )
    for key, val in data["validation"].items():
        frame[key] = np.asarray(val)
    keys = ["parent_id", "atmosphere_temperature_K", "atmosphere_pressure_bar"]
    baseline = frame.loc[frame["mode"] == "homogeneous"].set_index(keys)
    for species in ("H2", "H2O", "SiO", "SiH4"):
        baseline_values = (
            baseline[species]
            .reindex(pd.MultiIndex.from_frame(cast(pd.DataFrame, frame[keys])))
            .values
        )
        frame[f"{species}_ratio"] = frame[species] / baseline_values
    return frame


def control_error(frame, mask):
    ratios = frame.loc[mask, [f"{s}_ratio" for s in ("H2", "H2O", "SiO", "SiH4")]]
    return float(np.max(np.abs(np.log(ratios.to_numpy())))) if len(ratios) else 0.0


def save_figure(figure, name):
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(FIG / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_figures(deep_frame, atmospheres):
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#243747",
            "text.color": "#243747",
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
        }
    )
    model = SubregularSolution.hydrogen_silicate_rogers25()

    # Close the plotted curves at the independently calculated critical point,
    # rather than drawing a flat cap at the last finite temperature-grid cell.
    def instability_temperature(x):
        curvature = (
            2 * model.interaction_1
            - 4 * model.interaction_0
            + 6 * (model.interaction_0 - model.interaction_1) * x
        )
        return (
            -curvature
            * (1 + 10000 / model.pressure_scale)
            / (R / (x * (1 - x)) - curvature / model.temperature_scale)
        )

    critical = cast(
        OptimizeResult,
        minimize_scalar(
            lambda x: -instability_temperature(x), bounds=(0.34, 0.999), method="bounded"
        ),
    )
    assert critical.success
    critical_temperature = instability_temperature(critical.x)
    temperatures = np.linspace(2500, critical_temperature - 0.001, 240)
    diagnostic = model.diagnostics(temperatures, 10000.0, 0.5)
    figure, ax = plt.subplots(figsize=(8.3, 5.0), layout="constrained")
    lower, upper, sl, su = [
        np.append(np.asarray(diagnostic[k]), critical.x)
        for k in ("binodal_lower", "binodal_upper", "spinodal_lower", "spinodal_upper")
    ]
    temperatures = np.append(temperatures, critical_temperature)
    ax.fill_betweenx(temperatures, lower, upper, color="#deeeef", label="Two-phase equilibrium")
    ax.fill_betweenx(temperatures, sl, su, color="#edb780", label="Locally unstable (spinodal)")
    ax.plot(lower, temperatures, color=COLORS[0])
    ax.plot(upper, temperatures, color=COLORS[0])
    ax.plot(sl, temperatures, "--", color="#ab5a29")
    ax.plot(su, temperatures, "--", color="#ab5a29")
    for w, color in ((0.01, COLORS[2]), (0.03, COLORS[3])):
        x = (w / MOLAR_MASS["H2"]) / (w / MOLAR_MASS["H2"] + (1 - w) / MOLAR_MASS["MgSiO3"])
        ax.axvline(x, color=color, linewidth=1.5, linestyle=":")
        ax.text(x + 0.012, 4340, f"{100 * w:g} wt% H₂", color=color, va="top", fontsize=10)
    ax.set(
        xlim=(0, 1),
        ylim=(2500, 4400),
        xlabel="H₂ mole fraction on the H₂–MgSiO₃ component basis",
        ylabel="Deep temperature (K)",
        title="Equilibrium separation can precede spinodal instability · 1 GPa",
    )
    ax.legend(loc="lower left", framealpha=0.96, fontsize=10)
    save_figure(figure, "phase_diagram")

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), layout="constrained")
    for w, color in zip((1.0, 3.0, 10.0), COLORS):
        subset = deep_frame.loc[
            (deep_frame["pressure_bar"] == 10000) & (deep_frame["hydrogen_wt_percent"] == w)
        ]
        two_phase = np.asarray(subset["equilibrium_two_phase"], dtype=bool)
        # Single-phase output assigns all material to `lower`. Do not present
        # that labeling convention as physical loss of hydrogen or mass.
        axes[0].plot(
            subset["temperature_K"],
            np.where(two_phase, 100 * subset["upper_hydrogen_recovery"], np.nan),
            label=f"{w:g} wt% initial H₂",
            color=color,
        )
        axes[1].plot(
            subset["temperature_K"],
            np.where(two_phase, 100 * subset["upper_mass_kg"], np.nan),
            color=color,
        )
    axes[0].set(ylabel="Hydrogen in upper phase (% of total)", title="Hydrogen redistribution")
    axes[1].set(ylabel="Upper-phase mass (% of bulk)", title="The reservoir also changes size")
    for ax in axes:
        ax.set(xlabel="Deep temperature (K)", xlim=(2500, 4400))
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=9)
    figure.suptitle("Two-phase states at 1 GPa; curves stop outside coexistence", fontsize=12)
    save_figure(figure, "inventories")

    figure, axes = plt.subplots(2, 2, figsize=(9.6, 7.1), layout="constrained")
    for label, frame in atmospheres.items():
        condition = (
            (frame["deep_temperature_K"] == 3500)
            & (frame["hydrogen_wt_percent"] == 1)
            & (frame["atmosphere_temperature_K"] == 2500)
            & (frame["atmosphere_pressure_bar"] == 1)
        )
        separated = frame.loc[condition & (frame["mode"] == "spinodal")].sort_values(
            "entrained_lower_fraction"
        )
        baseline = frame.loc[condition & (frame["mode"] == "homogeneous")].iloc[0]
        color = COLORS[0] if label == "condensed" else COLORS[1]
        name = "With condensates" if label == "condensed" else "Gas-only control"
        for species, ax in zip(("H2", "H2O", "SiO", "SiH4"), axes.ravel()):
            ax.plot(
                separated["entrained_lower_fraction"],
                separated[species],
                "o-",
                color=color,
                label=name,
            )
            ax.axhline(baseline[species], color=color, alpha=0.5, linestyle=":")
            ax.set(
                xscale="symlog",
                yscale="log",
                title=species,
                xlabel="Lower-phase fraction entrained (η)",
                ylabel="Gas mole fraction",
                xticks=[0, 0.01, 0.1, 1],
                xticklabels=["0", "0.01", "0.1", "1"],
            )
            ax.set_xscale("symlog", linthresh=0.002)
            ax.set_xticks([0, 0.01, 0.1, 1], ["0", "0.01", "0.1", "1"])
            ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle(
        "1 wt% H₂ · 3500 K / 1 GPa source → 2500 K / 1 bar parcel\n"
        "Dotted lines: whole-bulk homogeneous-feed reference; connecting lines guide the eye",
        fontsize=12,
    )
    save_figure(figure, "entrainment")


def main():
    FIG.mkdir(exist_ok=True)
    deep = json.loads((OUT / "deep.json").read_text())
    frame = pd.DataFrame(deep["grid"])
    phase = deep["modes"]["spinodal"]["phase"]
    mixing = phase["mixing"]
    lower, upper = [phase["subphases"][s] for s in ("lower", "upper")]
    for key in ("spinodally_unstable", "metastable", "equilibrium_two_phase"):
        frame[key] = np.asarray(mixing[key]).ravel().astype(bool)
    frame["upper_mass_kg"] = array(upper["mass"])
    frame["upper_hydrogen_recovery"] = array(upper["species_number_moles"]["H2_fluid"]) / array(
        deep["modes"]["spinodal"]["species"]["number_moles"]["H2_fluid"]
    )
    frame["upper_silicate_recovery"] = array(upper["species_number_moles"]["MgO3Si_l"]) / array(
        deep["modes"]["spinodal"]["species"]["number_moles"]["MgO3Si_l"]
    )
    frame["gibbs_decrease_J_per_mol"] = array(
        mixing["homogeneous_mixing_gibbs_J_per_mol"]
    ) - array(mixing["actual_mixing_gibbs_J_per_mol"])
    frame["gibbs_decrease_J_per_kg_bulk"] = frame["gibbs_decrease_J_per_mol"] * array(
        phase["species_number_moles"]
    )
    conservation_error = 0.0
    for name, inventory in deep["modes"]["spinodal"]["species"]["number_moles"].items():
        recombined = array(lower["species_number_moles"][name]) + array(
            upper["species_number_moles"][name]
        )
        conservation_error = max(
            conservation_error, float(np.max(np.abs(recombined / array(inventory) - 1)))
        )
    assert conservation_error < 1e-10
    assert np.min(array(frame["gibbs_decrease_J_per_mol"])) > -1e-7
    assert all(all(m["validation"]["accepted"]) for m in deep["modes"].values())
    frame.to_csv(OUT / "deep_summary.csv", index=False)
    group = cast(
        pd.DataFrame,
        frame.groupby("hydrogen_wt_percent")[
            ["equilibrium_two_phase", "spinodally_unstable", "metastable"]
        ].sum(),
    )

    a, b, c, d = -6260.0, 786000.0, 4670.0, -350000.0
    x_limit = (2 * b - 4 * a) / (6 * (b - a))
    w_limit = (
        x_limit
        * MOLAR_MASS["H2"]
        / (x_limit * MOLAR_MASS["H2"] + (1 - x_limit) * MOLAR_MASS["MgSiO3"])
    )
    onset = []
    for w in (0.1, 0.3, 1.0, 3.0, 5.0, 10.0):
        x = (w / MOLAR_MASS["H2"]) / (w / MOLAR_MASS["H2"] + (100 - w) / MOLAR_MASS["MgSiO3"])
        excess_curvature = 2 * b - 4 * a + 6 * (a - b) * x
        temperature = (
            -excess_curvature * (1 + 10000 / d) / (R / (x * (1 - x)) - excess_curvature / c)
        )
        onset.append(
            {
                "hydrogen_wt_percent": w,
                "x_H2": x,
                "local_spinodal_onset_K_at_1_GPa": temperature if excess_curvature < 0 else None,
            }
        )

    atmospheres, atmospheric_summary, selected = {}, {}, {}
    for label in ("condensed", "gas_only"):
        data = json.loads((OUT / f"atmosphere_{label}.json").read_text())
        atmosphere = atmosphere_frame(data)
        assert np.asarray(atmosphere["accepted"]).all()
        remixed = (atmosphere["mode"] == "spinodal") & (
            atmosphere["entrained_lower_fraction"] == 1
        )
        unchanged = (atmosphere["mode"] == "spinodal") & ~atmosphere["separated"]
        remix_error, unchanged_error = [
            control_error(atmosphere, mask) for mask in (remixed, unchanged)
        ]
        assert max(remix_error, unchanged_error) < 1e-8
        selected[label] = atmosphere.loc[
            (atmosphere["deep_temperature_K"] == 3500)
            & (atmosphere["hydrogen_wt_percent"] == 1)
            & (atmosphere["atmosphere_temperature_K"] == 2500)
            & (atmosphere["atmosphere_pressure_bar"] == 1)
        ].to_dict(orient="records")
        pure = atmosphere.loc[
            (atmosphere["mode"] == "spinodal")
            & (atmosphere["entrained_lower_fraction"] == 0)
            & atmosphere["separated"]
        ]
        atmospheric_summary[label] = {
            "cases": len(atmosphere),
            "maximum_control_abs_log_ratio": max(remix_error, unchanged_error),
            "maximum_relative_element_error": float(
                np.max(array(atmosphere["maximum_relative_element_error"]))
            ),
            "maximum_abs_log_reaction_residual": float(
                np.max(array(atmosphere["maximum_abs_log_reaction_residual"]))
            ),
            "maximum_normalized_complementarity": float(
                np.max(array(atmosphere["maximum_normalized_complementarity"]))
            ),
            "pure_upper_phase_cases": len(pure),
            "pure_upper_phase_ratios": {
                s: {
                    "min": float(pure[f"{s}_ratio"].min()),
                    "max": float(pure[f"{s}_ratio"].max()),
                    "median": float(pure[f"{s}_ratio"].median()),
                }
                for s in ("H2", "H2O", "SiO", "SiH4")
            },
        }
        atmosphere.drop(
            columns=["available_component_moles", "normalized_component_mass_kg"]
        ).to_csv(OUT / f"atmosphere_{label}_summary.csv", index=False)
        atmospheres[label] = atmosphere
    selected_deep = (
        frame.loc[
            (frame["temperature_K"] == 3500)
            & (frame["pressure_bar"] == 10000)
            & (frame["hydrogen_wt_percent"] == 1)
        ]
        .iloc[0]
        .to_dict()
    )
    summary = {
        "deep_scenarios": len(frame),
        "deep_solves": 2 * len(frame),
        "deep_regime_counts": {
            k: int(np.sum(array(frame[k])))
            for k in ("equilibrium_two_phase", "spinodally_unstable", "metastable")
        },
        "deep_counts_by_hydrogen_wt_percent": group.reset_index().to_dict(orient="records"),
        "maximum_subphase_relative_component_error": conservation_error,
        "minimum_gibbs_decrease_J_per_mol": float(
            np.min(array(frame["gibbs_decrease_J_per_mol"]))
        ),
        "maximum_common_tangent_residual": float(np.max(array(mixing["common_tangent_residual"]))),
        "analytic_lower_spinodal_limit": {
            "x_H2": x_limit,
            "hydrogen_wt_percent": 100 * w_limit,
            "condition": "1-T/C+P/D >= 0; strictly positive T",
        },
        "analytic_spinodal_onsets": onset,
        "selected_deep_case": selected_deep,
        "atmosphere": atmospheric_summary,
        "selected_atmospheric_cases": selected,
    }
    save(OUT / "summary.json", summary)
    make_figures(frame, atmospheres)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
