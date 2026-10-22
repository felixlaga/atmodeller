# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interface for initializers of the solver's initial guess

An initializer provides the initial guess, the concatenated array of [log_number_moles,
log_stability], from which the solver starts. It separates the first (cold) guess, from
:meth:`InitializerProtocol.start`, from subsequent (warm) guesses, from
:meth:`InitializerProtocol.iterate`, which may use a previous solution. Rows of the previous
solution that contain ``NaN`` have no previous solution.

Both methods receive the vmap axes of the parameters, as computed by
:func:`~atmodeller.jax_utils.vmap_axes_spec` when the solver is built. An initializer that
solves each system separately needs them to vmap over a batch, since they cannot be recovered
from the parameters under JIT (NumPy leaves are then traced too). If ``None``, they are computed
from the parameters, which is only valid outside JIT.

Concrete implementations are in :mod:`atmodeller.initializer`.
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float, PyTree

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

    def start(
        self, parameters: Parameters, vmap_axes: PyTree | None = None
    ) -> Float[Array, "... twice_species"]:
        """Cold initial guess that does not use a previous solution.

        Args:
            parameters: Parameters
            vmap_axes: vmap axes of ``parameters``. Defaults to ``None``, which computes them
                from ``parameters`` (only valid outside JIT).

        Returns:
            Concatenated array of [log_number_moles, log_stability]
        """
        ...

    def iterate(
        self,
        parameters: Parameters,
        previous: Float[Array, "... twice_species"],
        vmap_axes: PyTree | None = None,
    ) -> Float[Array, "... twice_species"]:
        """Initial guess given a previous solution.

        Args:
            parameters: Parameters
            previous: Previous solution. Rows with ``NaN`` have no previous solution.
            vmap_axes: vmap axes of ``parameters``. Defaults to ``None``, which computes them
                from ``parameters`` (only valid outside JIT).

        Returns:
            Concatenated array of [log_number_moles, log_stability]
        """
        ...
