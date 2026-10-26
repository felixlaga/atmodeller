# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Equilibrium model API

Provides :class:`EquilibriumModel`, the user-facing entry point for running thermodynamic
equilibrium calculations from a prepared :class:`~atmodeller.parameters.Parameters` object.

The module couples immutable model parameters to a compiled solver callable, then exposes small
wrapper methods for solving and parameter updates while preserving Equinox tree semantics.

Typical usage:

.. code-block:: python

    from atmodeller.classes import EquilibriumModel
    from atmodeller.parameters import Parameters

    # Create Parameters object (not shown)
    parameters = Parameters(...)
    model = EquilibriumModel(parameters)
    output = model.solve_with_default()
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from atmodeller.containers import ChemicalSpecies, SolverParameters
from atmodeller.initializer import ElementPotentialInitializer, InitializerProtocol
from atmodeller.interfaces import ActivityConstraintProtocol, SpeciesProtocol
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import PhaseEquilibriumMode, PurePhase
from atmodeller.solvers import make_solver_with_jit
from atmodeller.state import BaseThermodynamicState, ThermodynamicState
from atmodeller.thermodata.mixing import SubregularSolution

logger: logging.Logger = logging.getLogger(__name__)

SolverCallable = Callable[
    [Parameters, PRNGKeyArray, Float[Array, "n_batch twice_species"]], Output
]


class EquilibriumModel(eqx.Module):
    """Main user-facing API to run equilibrium calculations

    The model bundles immutable :class:`~atmodeller.parameters.Parameters`, an initializer for
    the solver's initial guess, a compiled solver closure, and a random key used by stochastic
    solver routines.

    Args:
        parameters: Fully prepared model parameters, including state, constraints, and solver
            settings.
        initializer: Initializer for the cold initial guess, used for batch elements without an
            initial guess. Defaults to
            :class:`~atmodeller.initializer.element_potential.ElementPotentialInitializer`.
        phase_equilibrium: Optional ``"spinodal"`` or ``"homogeneous"`` binary-solution mode.
            ``None`` preserves the state's existing phase settings.
    """

    parameters: Parameters
    initializer: InitializerProtocol = eqx.field(static=True)
    solver: SolverCallable
    key: PRNGKeyArray

    def __init__(
        self,
        parameters: Parameters,
        initializer: InitializerProtocol | None = None,
        *,
        phase_equilibrium: PhaseEquilibriumMode | None = None,
    ):
        if phase_equilibrium is not None:
            parameters = parameters.update_state(phase_equilibrium=phase_equilibrium)
        self.parameters = parameters
        self.initializer = ElementPotentialInitializer() if initializer is None else initializer
        self.solver = make_solver_with_jit(parameters, self.initializer)
        self.key = jax.random.PRNGKey(0)

    @classmethod
    def from_state(
        cls,
        state: BaseThermodynamicState,
        *,
        activity_constraints: Mapping[str, ActivityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        mass_units: Literal["mass", "moles"] = "mass",
        solver_parameters: SolverParameters | None = None,
        initializer: InitializerProtocol | None = None,
        phase_equilibrium: PhaseEquilibriumMode | None = None,
    ) -> Self:
        """Builds a model by constructing :class:`~atmodeller.parameters.Parameters` internally.

        This is a convenience constructor equivalent to creating ``Parameters(...)`` first and
        then passing that object to :class:`EquilibriumModel`.

        Args:
            state: Thermodynamic state
            activity_constraints: Optional activity/fugacity constraint mapping. Defaults to
                ``None``.
            mass_constraints: Optional mass-constraint mapping. Defaults to ``None``.
            mass_units: Units used for ``mass_constraints``. Defaults to ``"mass"``.
            solver_parameters: Optional solver parameter object. Defaults to ``None``.
            initializer: Optional initializer for the cold initial guess. Defaults to ``None``.
            phase_equilibrium: Optional binary-solution equilibrium mode; ``None`` preserves
                the state's settings. Spinodal mode requires a configured binary mixing model.

        Returns:
            A new :class:`EquilibriumModel` instance
        """
        parameters: Parameters = Parameters(
            state=state,
            activity_constraints=activity_constraints,
            mass_constraints=mass_constraints,
            mass_units=mass_units,
            solver_parameters=solver_parameters,
        )

        return cls(parameters, initializer, phase_equilibrium=phase_equilibrium)

    @classmethod
    def from_species(
        cls,
        gas_species: Iterable[ChemicalSpecies],
        pressure: ArrayLike,
        *,
        temperature: ArrayLike = 2000.0,
        silicate_melt_species: Iterable[SpeciesProtocol] = (),
        silicate_solid_species: Iterable[SpeciesProtocol] = (),
        metal_species: Iterable[SpeciesProtocol] = (),
        condensates: Iterable[PurePhase] = (),
        mixing_models: Mapping[str, SubregularSolution] | None = None,
        phase_equilibrium: PhaseEquilibriumMode | None = None,
        activity_constraints: Mapping[str, ActivityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        mass_units: Literal["mass", "moles"] = "mass",
        solver_parameters: SolverParameters | None = None,
        initializer: InitializerProtocol | None = None,
        **state_options: Any,
    ) -> Self:
        """Builds a fixed-pressure model from temperature, composition and phase treatment.

        This combines ``ThermodynamicState.from_species`` and ``from_state``;
        the same equilibrium engine and conservation equations are used.
        For a planetary pressure law, construct ``Planet.from_species`` with
        the same ``mixing_models``/``phase_equilibrium`` options, then use ``from_state``.

        Args:
            gas_species: Gas species.
            pressure: Pressure (bar).
            temperature: Temperature (K).
            silicate_melt_species: Components of the silicate melt.
            silicate_solid_species: Components of the silicate solid.
            metal_species: Components of the metal phase.
            condensates: Additional pure condensates.
            mixing_models: Binary mixing free energies keyed by condensed phase name.
            phase_equilibrium: ``"spinodal"`` (common-tangent equilibrium) or
                ``"homogeneous"``. ``None`` keeps the phase defaults.
            activity_constraints: Optional activity constraints.
            mass_constraints: Elemental or compound inventories, in ``mass_units``.
            mass_units: ``"mass"`` (kg) or ``"moles"`` (mol of each specified formula).
            solver_parameters: Numerical solver settings.
            initializer: Initial-guess strategy.
            **state_options: Other ``ThermodynamicState.from_species`` arguments,
                such as ``mass=0`` for fully tracked phases with no inert solvent.
        """
        state = ThermodynamicState.from_species(
            gas_species,
            pressure,
            temperature=temperature,
            silicate_melt_species=silicate_melt_species,
            silicate_solid_species=silicate_solid_species,
            metal_species=metal_species,
            condensates=condensates,
            mixing_models=mixing_models,
            phase_equilibrium=phase_equilibrium,
            **state_options,
        )
        return cls.from_state(
            state,
            activity_constraints=activity_constraints,
            mass_constraints=mass_constraints,
            mass_units=mass_units,
            solver_parameters=solver_parameters,
            initializer=initializer,
        )

    def with_phase_equilibrium(self, mode: PhaseEquilibriumMode) -> Self:
        """Returns a model with a new binary equilibrium mode and rebuilt solver.

        The original model, inventories, thermodynamic functions and random key
        are preserved. Use this for matched homogeneous/spinodal comparisons.
        """
        return self.update_state(phase_equilibrium=mode)

    def solve(self, base_solution_array: Float[Array, "n_batch twice_species"]) -> Output:
        """Runs the solver and returns the output state.

        Note:
            This method is intentionally a thin Python wrapper. The heavy numerical path is
            compiled inside ``self.solver``
            (created by :func:`~atmodeller.solvers.make_solver_with_jit`).
            The same key is reused for repeat calls, so stochastic branches remain deterministic.

        Args:
            base_solution_array: Initial guess for the solver, typically a broadcasted array of
                log number moles and log stabilities. Rows are used as given, and rows that
                contain ``NaN`` are replaced by the cold guess of the model's initializer.

        Returns:
            An :class:`~atmodeller.output.Output` instance
        """
        self._log_initial_guess(base_solution_array)

        return self.solver(self.parameters, self.key, base_solution_array)

    def _log_initial_guess(self, base_solution_array: ArrayLike) -> None:
        """Logs how many models start from the initializer's cold guess and how many from the
        given initial guess. Nothing is logged if the initial guess is traced (inside JIT).

        Args:
            base_solution_array: Initial guess for the solver
        """
        try:
            guess: np.ndarray = np.atleast_2d(np.asarray(base_solution_array))
        except jax.errors.TracerArrayConversionError:
            return

        batch_size: int = self.parameters.batch_size
        cold: np.ndarray = np.broadcast_to(np.any(np.isnan(guess), axis=-1), (batch_size,))
        logger.info(
            "Initial guess: %d model(s) from %s, %d model(s) from the given initial guess",
            np.count_nonzero(cold),
            type(self.initializer).__name__,
            batch_size - np.count_nonzero(cold),
        )

    def solve_with_default(self) -> Output:
        """Runs the solver with a default initial guess.

        Note:
            Like :meth:`solve`, this method is a lightweight non-jitted wrapper around the
            already-jitted ``self.solver`` callable.

        Returns:
            An :class:`~atmodeller.output.Output` instance
        """
        base_solution_array: Float[Array, "n_batch twice_species"] = jnp.full(
            (
                self.parameters.batch_size,
                self.parameters.reaction_system.species.number_species * 2,
            ),
            jnp.nan,
        )

        return self.solve(base_solution_array)

    def rebuild_solver(self) -> Self:
        """Rebuilds the compiled solver from the model's current parameters.

        Use this after parameter updates that may have changed array-leaf shapes or batching in a
        way that invalidates the vmapping axes captured by the existing solver closure, or that
        invalidates the original leaf-shape broadcasting assumptions preserved by the
        ``update`` methods on the parameter containers.

        Returns:
            A new instance of :class:`EquilibriumModel` with a rebuilt solver
        """
        solver_rebuilt: SolverCallable = make_solver_with_jit(self.parameters, self.initializer)
        model_rebuilt: EquilibriumModel = eqx.tree_at(lambda m: m.solver, self, solver_rebuilt)

        return cast(Self, model_rebuilt)

    def update_constraints(
        self,
        *,
        activity_constraints: Mapping[str, ActivityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        mass_units: Literal["mass", "moles"] = "mass",
    ) -> Self:
        """Returns a model with updated mass and activity/fugacity constraints.

        This is a convenience wrapper around
        :meth:`atmodeller.parameters.Parameters.update_constraints` that preserves the immutable
        Equinox tree semantics of the model.

        Args:
            activity_constraints: New activity/fugacity constraint mapping. Defaults to ``None``.
            mass_constraints: New mass-constraint mapping. Defaults to ``None``.
            mass_units: Units used for ``mass_constraints``. Defaults to ``"mass"``.

        Returns:
            A new model instance with updated parameters
        """
        parameters_updated: Parameters = self.parameters.update_constraints(
            activity_constraints=activity_constraints,
            mass_constraints=mass_constraints,
            mass_units=mass_units,
        )
        model_updated: EquilibriumModel = eqx.tree_at(
            lambda m: m.parameters, self, parameters_updated
        )

        return cast(Self, model_updated)

    def update_state(self, *args: object, **kwargs: object) -> Self:
        """Returns a model with an updated thermodynamic state.

        Arguments are forwarded to ``self.parameters.update_state``, which in turn forwards them
        to the underlying thermodynamic state's ``update`` implementation.
        A ``phase_equilibrium`` change also rebuilds the solver, since it changes
        the static phase treatment. Temperature/pressure updates retain the
        existing shape-preserving behavior.

        Args:
            *args: Positional arguments accepted by the state ``update`` method
            **kwargs: Keyword arguments accepted by the state ``update`` method

        Returns:
            A new model instance with updated parameters
        """
        parameters_updated: Parameters = self.parameters.update_state(*args, **kwargs)
        model_updated: EquilibriumModel = eqx.tree_at(
            lambda m: m.parameters, self, parameters_updated
        )

        if kwargs.get("phase_equilibrium") is not None:
            model_updated = model_updated.rebuild_solver()
        return cast(Self, model_updated)
