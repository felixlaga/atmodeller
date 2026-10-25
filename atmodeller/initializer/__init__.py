# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Initializers for the solver's initial guess

An initializer provides the cold initial guess for a single system, and is vmapped over the batch
by the solver (see :func:`~atmodeller.solvers.make_initial_guess_function`).

- :class:`~atmodeller.initializer.core.InitializerProtocol`: The interface
- :class:`~atmodeller.initializer.element_potential.ElementPotentialInitializer`: Solves an
  ideal-gas equilibrium in the element potentials; the default
- :class:`~atmodeller.initializer.heuristic.HeuristicInitializer`: Allocates element budgets
  heuristically
"""

from atmodeller.initializer.core import InitializerProtocol  # noqa: F401
from atmodeller.initializer.element_potential import ElementPotentialInitializer  # noqa: F401
from atmodeller.initializer.heuristic import HeuristicInitializer  # noqa: F401
