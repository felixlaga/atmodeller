# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Interface for initializers of the solver's initial guess

An initializer provides the cold initial guess, the concatenated array of [log_number_moles,
log_stability], from which the solver starts when no previous solution is available. It acts on a
single system and is vmapped over the batch by the solver, in the same way as
:func:`~atmodeller.solvers.solve_single`.

The solver decides which batch elements need a cold guess: rows of the initial guess passed to the
solver are used as given, and rows that contain ``NaN`` are replaced by the initializer's guess.
If no row contains ``NaN``, the initializer is not evaluated.

Concrete implementations are in :mod:`atmodeller.initializer`.
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float

from atmodeller.parameters import Parameters


@runtime_checkable
class InitializerProtocol(Protocol):
    """Protocol for an initializer that provides the cold initial guess for the solver

    Implementations should be stateless :class:`equinox.Module` instances whose settings are plain
    Python numbers, which are constant under JIT, so reusing an initializer never triggers
    retracing.
    """

    def start(self, parameters: Parameters) -> Float[Array, " twice_species"]:
        """Cold initial guess for a single system.

        Args:
            parameters: Parameters of a single system

        Returns:
            Concatenated array of [log_number_moles, log_stability]
        """
        ...
