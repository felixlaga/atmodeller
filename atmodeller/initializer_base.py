# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interface for initializers of the solver's initial guess

An initializer provides the initial guess, the concatenated array of [log_number_moles,
log_stability], from which the solver starts. It separates the first (cold) guess, from
:meth:`InitializerProtocol.start`, from subsequent (warm) guesses, from
:meth:`InitializerProtocol.iterate`, which may use a previous solution. Rows of the previous
solution that contain ``NaN`` have no previous solution.

Concrete implementations are in :mod:`atmodeller.initializer`.
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float

from atmodeller.parameters import Parameters


@runtime_checkable
class InitializerProtocol(Protocol):
    """Protocol for an initializer that provides the initial guess for the solver

    An initializer separates the first (cold) guess from subsequent (warm) guesses. Whether a
    previous solution is available is determined by the data rather than by stored state: rows of
    ``previous`` that contain ``NaN`` have no previous solution. Implementations should be
    stateless :class:`equinox.Module` instances without array fields so that they are static
    under JIT and reusing an initializer never triggers retracing.
    """

    def start(self, parameters: Parameters) -> Float[Array, "... twice_species"]:
        """Cold initial guess that does not use a previous solution.

        Args:
            parameters: Parameters

        Returns:
            Concatenated array of [log_number_moles, log_stability]
        """
        ...

    def iterate(
        self, parameters: Parameters, previous: Float[Array, "... twice_species"]
    ) -> Float[Array, "... twice_species"]:
        """Initial guess given a previous solution.

        Args:
            parameters: Parameters
            previous: Previous solution. Rows with ``NaN`` have no previous solution.

        Returns:
            Concatenated array of [log_number_moles, log_stability]
        """
        ...
