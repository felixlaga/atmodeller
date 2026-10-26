# SPDX-License-Identifier: Apache-2.0

"""Run the first binary-demixing examples using the native Atmodeller interfaces.

From the repository root: uv run python examples/spinodal_equilibrium.py

The hydrogen-silicate calculation evaluates a published mixing free energy.
The gas-coupling example uses Fe/Si species and an *illustrative*, uncalibrated
regular-solution interaction to exercise the equilibrium engine. It is not a
prediction of real Fe-Si immiscibility or of a sub-Neptune atmosphere.
"""

import json

import numpy as np

from atmodeller import (
    ChemicalSpecies,
    EquilibriumModel,
    PhaseEquilibriumMode,
    SolverParameters,
    SubregularSolution,
)
from atmodeller.sci_utils import GAS_CONSTANT


def gas_solution_example(
    interaction: float, *, phase_equilibrium: PhaseEquilibriumMode = "spinodal"
) -> dict:
    """Compare ideal, constrained homogeneous, and demixed solution activities."""
    components = tuple(ChemicalSpecies.create_solution(s) for s in ("Fe", "Si"))
    masses = {
        "He": 0.004002602,
        "Fe": 0.75 * components[0].data.molar_mass,
        "Si": 0.25 * components[1].data.molar_mass,
    }
    model = EquilibriumModel.from_species(
        [ChemicalSpecies.create_gas(s) for s in ("He", "Fe", "Si")],
        temperature=2000.0,
        pressure=1.0,
        mass=0.0,
        metal_species=components,
        mixing_models={"metal": SubregularSolution.regular_solution(interaction)},
        phase_equilibrium=phase_equilibrium,
        mass_constraints=masses,
        solver_parameters=SolverParameters(atol=1e-10, rtol=1e-10),
    )
    output = model.solve_with_default()
    if not np.all(output.multi_attempt_solution.success):
        raise RuntimeError("The example did not converge.")
    data = output.to_dict(to_numpy=True)
    errors = {}
    for element, target in masses.items():
        total = sum(
            np.asarray(data[p]["elements"]["mass"].get(element, 0)).sum() for p in ("gas", "metal")
        )
        errors[element] = float(abs(total - target) / target)
    if max(errors.values()) > 1e-9:
        raise RuntimeError("The example did not conserve its elements.")
    return {
        "partial_pressure_bar": {
            k: float(v.item()) for k, v in data["gas"]["species"]["partial_pressure"].items()
        },
        "mixing": {k: v.item() for k, v in data["metal"]["phase"]["mixing"].items()},
        "relative_element_errors": errors,
    }


def main() -> None:
    hydrogen_silicate = SubregularSolution.hydrogen_silicate_rogers25()
    diagnostics = hydrogen_silicate.diagnostics(3975.0, 10000.0, 0.6)
    results = {
        "hydrogen_silicate_at_3975K_1GPa_xH2_0p6": {k: v.item() for k, v in diagnostics.items()},
        "illustrative_gas_solution": {
            "ideal": gas_solution_example(0),
            "constrained_homogeneous": gas_solution_example(
                3 * GAS_CONSTANT * 2000, phase_equilibrium="homogeneous"
            ),
            "equilibrium_demixing": gas_solution_example(3 * GAS_CONSTANT * 2000),
        },
    }

    # Undefined phase boundaries (no gap) are null in portable JSON.
    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        return None if isinstance(value, float) and not np.isfinite(value) else value

    print(json.dumps(portable(results), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
