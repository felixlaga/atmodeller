"""Hakim-style v2 Atmodeller model and explicit solubility sensitivity.

No phase-separation or spinodal free energy is implemented here.
All masses are kg, temperatures K, pressures/fugacities bar.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import lineax as lx
from pathlib import Path
from atmodeller import (
    ChemicalSpecies, ReservoirSpecies, Planet, PurePhase,
    EquilibriumModel, SolverParameters, get_solubility_models,
)
from atmodeller.eos import get_eos_models
from atmodeller.solubility.core import Solubility

jax.config.update("jax_compilation_cache_dir", str(Path(__file__).resolve().parent / "outputs" / ".jax_cache"))

PLANET_MASS = 6.7 * 5.972e24
MEB_RADIUS = 1.65 * 6371000
ARCHIVE_CORE_FRACTION = 0.295334691460966
SILICON_FRACTION = round(45.4 / 98.41 * (1 - 0.327) * 28.0855 / (28.0855 + 2 * 15.999), 3)
OXYGEN_FRACTION = SILICON_FRACTION * 2 * 15.999 / 28.0855
GASES = ("H2O", "H2", "O2", "OSi", "H4Si", "CO2", "CO", "CH4", "N2", "H3N", "He")
EOS = dict(zip(GASES, (
    "H2O_cork_holland98", "H2_chabrier21", "O2_cs_shi92", "OSi_rk49_connolly16",
    "H4Si_wang18", "CO2_cs_shi92", "CO_cs_shi92", "CH4_cs_shi92", "N2_cs_saxena87",
    "H3N_rk49_reid87", "He_chabrier21",
)))
SOLUBILITY = {
    "H2O": "H2O_peridotite_sossi23", "H2": "H2_basalt_hirschmann12",
    "CO2": "CO2_basalt_dixon95", "CO": "CO_basalt_yoshioka19",
    "CH4": "CH4_basalt_ardia13", "N2": "N2_basalt_libourel03", "He": "He_basalt_jambon86",
}


class ScaledSolubility(Solubility):
    """Multiply a published dissolution law; multiplier is a scenario, not fitted physics."""
    base: Solubility
    multiplier: object = eqx.field(converter=lambda v: jnp.asarray(v, dtype=jnp.float64))

    def concentration(self, fugacity, *, temperature=None, pressure=None, fO2=None):
        return self.multiplier * self.base.concentration(
            fugacity, temperature=temperature, pressure=pressure, fO2=fO2
        )


def budgets(hydrogen_wt_percent=1.0, metallicity=1.0):
    h = np.asarray(hydrogen_wt_percent) / 100 * PLANET_MASS
    z = np.asarray(metallicity)
    return {
        "H": h, "He": h * 24.69 / 73.9,
        "C": h * 0.22 / 73.9 * z, "N": h * 0.07 / 73.9 * z,
        "Si": h * 0.07 / 73.9 * z + SILICON_FRACTION * PLANET_MASS,
        "O": h * 0.63 / 73.9 * z + OXYGEN_FRACTION * PLANET_MASS,
    }


def make_gases(real=True):
    eos = get_eos_models()
    return tuple(ChemicalSpecies.create_gas(s, **({"activity": eos[EOS[s]]} if real else {})) for s in GASES)


def make_meb(hydrogen_wt_percent=1.0, metallicity=1.0, temperature=3000.0,
             h2_scale=1.0, h2o_scale=1.0, pressure=np.nan, real=True, tolerance=1e-8, method="newton"):
    sol = get_solubility_models()
    dissolved = tuple(ReservoirSpecies.create_dissolved(
        s, solubility=ScaledSolubility(sol[law], h2_scale if s == "H2" else h2o_scale if s == "H2O" else 1.0),
        include_in_phase_mass=False,
    ) for s, law in SOLUBILITY.items())
    state = Planet.from_species(
        make_gases(real), silicate_melt_species=dissolved,
        condensates=(PurePhase.from_species("O2Si", state="l"),),
        planet_mass=PLANET_MASS, core_mass_fraction=ARCHIVE_CORE_FRACTION,
        surface_radius=MEB_RADIUS, temperature=temperature, pressure=pressure,
    )
    mass = budgets(hydrogen_wt_percent, metallicity)
    numerical = dict(solver=optx.LevenbergMarquardt, linear_solver=lx.QR(), max_steps=1024) if method == "lm" else {}
    return EquilibriumModel.from_state(state, mass_constraints=mass,
        solver_parameters=SolverParameters(atol=tolerance, rtol=tolerance, **numerical)), mass


def make_atmosphere(mass, pressure, temperature=920.0, condensates=True, tolerance=1e-8, method="newton"):
    phases = tuple(PurePhase.from_species(s, state="s") for s in ("O2Si", "C", "CSi", "Si", "N4Si3")) if condensates else ()
    state = Planet.from_species(make_gases(False), condensates=phases,
        planet_mass=PLANET_MASS, surface_radius=2.64 * 6371000,
        mantle_melt_fraction=0, pressure=pressure, temperature=temperature)
    numerical = dict(solver=optx.LevenbergMarquardt, linear_solver=lx.QR(), max_steps=1024) if method == "lm" else {}
    return EquilibriumModel.from_state(state, mass_constraints=mass,
        solver_parameters=SolverParameters(atol=tolerance, rtol=tolerance, **numerical))


def serializable(v):
    if isinstance(v, dict):
        return {k: serializable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [serializable(x) for x in v]
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    return np.asarray(v).tolist()


def extract(output, mass):
    data = output.to_dict(to_numpy=True)
    gas = data["gas"]
    n = output.parameters.batch_size
    def arr(x):
        return np.broadcast_to(np.asarray(x).reshape(-1), (n,))
    phase_names = [p.name for p in output.parameters.state.phase_system.phases if not p.is_empty]
    phase_mass = {p: {e: arr(v) for e, v in data[p]["elements"]["mass"].items()}
                  for p in phase_names}
    errors = {e: (sum(d.get(e, 0.0) for d in phase_mass.values()) - arr(v)) / arr(v)
              for e, v in mass.items()}
    partial = {s: arr(v) for s, v in gas["species"]["partial_pressure"].items()}
    return {
        "success": arr(output.multi_attempt_solution.success).astype(bool),
        "pressure_bar": sum(partial.values()),
        "mole_fraction": {s: arr(v) for s, v in gas["species"]["mole_fraction"].items()},
        "partial_pressure_bar": partial,
        "fugacity_bar": {s: arr(v) for s, v in gas["species"]["activity"].items()},
        "element_phase_mass_kg": phase_mass,
        "relative_element_errors": errors,
        "phase_species_mass_kg": {p: {s: arr(v) for s, v in d["species"]["mass"].items()}
                                  for p, d in data.items() if p in phase_mass},
        "mass_constraints_kg": {e: arr(v) for e, v in mass.items()},
    }
