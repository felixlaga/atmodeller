Binary solutions and spinodal stability
=======================================

This fork adds a binary mixing free energy, its spinodal and binodal, and
equilibrium demixing through the existing phase/activity interface. It is the
thermodynamic foundation for a spinodal-decomposition model. It does not yet
evolve composition in space or time.

The earlier hydrogen-storage investigation is preserved under
``research/hydrogen_storage/``. Those results used hypothetical solubility
multipliers and the unmodified upstream engine. They are not results of this
extension. See ``research/README.rst`` for provenance and reproduction.

Thermodynamic model
-------------------

:class:`~atmodeller.thermodata.mixing.SubregularSolution` uses an ordered pair
of components, with :math:`x=n_1/(n_0+n_1)`. All energies are per mole of those
components, temperature is K, and pressure is bar:

.. math::

   g_{\rm mix}=RT[(1-x)\ln(1-x)+x\ln x]
      +x(1-x)[A(1-x)+Bx](1-T/C+P/D).

Infinite C or D disables the respective dependence. Setting A=B and disabling
both dependences recovers the symmetric regular solution. The homogeneous
partial molar mixing energies are calculated from this same function:

.. math::

   \mu_0^{\rm mix}=g_{\rm mix}-xg'_{\rm mix},\qquad
   \mu_1^{\rm mix}=g_{\rm mix}+(1-x)g'_{\rm mix},\qquad
   \ln a_i=\mu_i^{\rm mix}/RT.

The pure-component standard Gibbs energies remain in ``ChemicalSpecies.thermo``.
They must use the same component basis and compatible standard states. Do not
add a solubility correction to the mixing activities without deriving the
combined free energy and its reference states.

* **Spinodal:** homogeneous :math:`g''_{\rm mix}<0`; infinitesimal composition
  fluctuations lower free energy. The limits solve :math:`g''_{\rm mix}=0`.
* **Binodal:** the compositions of two coexisting phases, found by equal chemical
  potentials for both components (a common tangent).
* **Metastable:** inside the binodal but outside the spinodal. A finite
  fluctuation can lower free energy even though the local curvature is positive.

The equilibrium activity uses the common tangent throughout the binodal gap,
including the metastable part. Separation is therefore not triggered only when
the spinodal is crossed. For bulk composition x between endpoints
:math:`x_\alpha,x_\beta`, the fraction of component moles in the upper-composition
phase is :math:`f_\beta=(x-x_\alpha)/(x_\beta-x_\alpha)`. This is not a mass
fraction. The two component inventories are recovered by the lever rule.

Hydrogen-silicate parameterization
----------------------------------

``SubregularSolution.hydrogen_silicate_rogers25()`` implements equation 7 of
`Rogers, Young & Schlichting (2025)
<https://doi.org/10.1093/mnras/staf1940>`_ (the explicit equation and parameters are
also in `arXiv:2509.13320v1 <https://arxiv.org/html/2509.13320v1#S2.SS1.SSS1>`_).
Component 0 is MgSiO3 and component 1 is H2. A=-6260 and B=786000 J/mol,
C=4670 K, and D=-35 GPa, converted here to -350000 bar.

This is the published phenomenological fit, not a new fit to the final 2026
DFT results, nor a calibration for water-silicate, hydrogen-helium, or a
multicomponent magma. A finite numerical result does not establish that a
parameterization is valid outside its source's physical regime.

For example::

    from atmodeller import SubregularSolution

    mixing = SubregularSolution.hydrogen_silicate_rogers25()
    result = mixing.diagnostics(3975.0, 10000.0, 0.6)
    coexistence = mixing.coexistence(3975.0, 10000.0)

At these inputs the implementation gives binodal H2 mole fractions
0.4349126578 and 0.9469194554, and spinodal limits 0.5721591077 and 0.8787976303.
The bulk composition 0.6 is locally unstable and the equilibrium upper-phase
mole fraction is 0.3224319344. These numbers describe that mixing model and do
not by themselves give atmospheric molecular abundances.

Using the native equilibrium engine
-----------------------------------

:class:`~atmodeller.phases.BinarySolutionPhase` subclasses ``CondensedPhase``.
It overrides the same ``get_log_activity`` method used by the reaction engine,
with no new residual block, duplicate component inventory or separate solver.
Its ``BinarySolutionPhaseOutput`` follows the existing phase/output split and
adds diagnostics at ``output[phase_name]["phase"]["mixing"]`` in all dictionary
formats and DataFrame exports.

``spinodally_unstable`` and ``metastable`` describe the hypothetical homogeneous
mixture at the bulk composition. A stable, demixed equilibrium can therefore
report ``spinodally_unstable=True``: its two actual coexisting compositions lie
on the stable branches. ``demixing_enabled`` records which activities were used.

Create its two components with ``ChemicalSpecies.create_solution``. Their
amounts are solved, their masses count in the phase, and they have no separate
pure-component stability variables. A binary solution cannot contain an inert
background, reservoir/solubility species, or additional independent activity
models. The constructor rejects inconsistent component configurations.

The phase assumes the solution remains present. Collective appearance or
disappearance of the entire solution phase is not implemented. Check that the
solution has a finite inventory in the intended physical regime; the pure-phase
stability variables are not a substitute for this missing collective condition.

Use ``PhaseSystem`` and ``ReactionSystem`` to assemble it into a state. Preserve
the engine's phase order: gas, silicate melt, silicate solid, metal, then any
additional phases. The example below uses empty phases as placeholders. A
planetary state can likewise be built with a reaction system containing this
phase; no change to the elemental conservation equations is needed.

Run the complete example from the repository root::

    uv sync --locked --all-extras --dev
    uv run python examples/spinodal_equilibrium.py

.. literalinclude:: ../examples/spinodal_equilibrium.py
   :language: python
   :pyobject: gas_solution_example

The example uses database Fe/Si species and an **illustrative, uncalibrated**
regular-solution interaction. It verifies gas coupling; it does not predict real
Fe-Si immiscibility. At 2000 K and 1 bar its ideal-solution Si partial pressure
is 1.08830e-5 bar, compared with 4.10549e-5 bar after equilibrium demixing with
Omega=3RT. Each elemental inventory is conserved. Setting ``equilibrate=False``
computes the constrained homogeneous case, which can be unstable; numerical
convergence is not a claim of stable physical equilibrium.

Numerics and differentiability
------------------------------

The subregular curvature has one minimum because its fourth derivative is
positive. Bracketed searches find the spinodal limits, then the common tangent
between the two stable branches. This excludes the spurious equal-composition
solution of unconstrained chemical-potential matching. Log-odds coordinates
resolve exponentially dilute coexisting components.

Equinox modules retain the model parameters in their pytrees. Array inputs
broadcast elementwise; the methods support JIT and vmap. Implicit derivatives
of the spinodal and common-tangent equations are used instead of differentiating
through bisection decisions. The two-phase derivatives are singular at a critical
point and phase-boundary derivatives are branch-dependent. Do not assume global
smoothness across changes of phase regime.

``BinaryCoexistence.valid`` checks finite inputs, the separated branch brackets,
log odds within +/-699 (the search interval is +/-700), and chemical-potential
mismatch below 1e-8 in mu/RT. An invalid tangent gives NaN activities, rather
than reverting to a homogeneous state. When there is no gap, ``has_gap=False``;
the raw coexistence object uses equal-composition sentinels and diagnostics
report NaN boundaries. Always inspect these flags. Spinodal searches use
composition bounds 1e-14 and 1-1e-14; extremely large interaction parameters
outside this numerical range are not supported.

Tests check analytic ideal/regular limits, Gibbs-Duhem consistency, a global
supporting tangent, independent SciPy coexistence roots, component conservation,
component reversal, pressure conversion, JIT/vmap, derivatives, dilute and
near-critical cases, invalid calculations, native gas equilibrium and exports.

What is needed for an atmospheric prediction
--------------------------------------------

The next physical step is to reconcile the H2-MgSiO3 free energy and both
endmember standard states with the reacting atmospheric network. An H2-rich
branch of this binary model is not automatically an independently speciated
ideal-gas atmosphere; counting both as the same hydrogen reservoir would be
inconsistent. Partitioning of additional elements needs a multicomponent
free energy or independently justified transfer chemical potentials.

A time-dependent spinodal model additionally needs a gradient-energy coefficient
and a mobility/diffusion model, with a consistent density or molar-volume
convention. Transport, settling and cooling then determine whether the
equilibrium end state is reached. None of those material parameters are inferred
from the binodal or supplied here. The present code supplies thermodynamic
driving forces and a validated equilibrium limit for that later development.
