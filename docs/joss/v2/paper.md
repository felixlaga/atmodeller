---
# Use plain text in the title; do not use backticks for code formatting.
title: 'Atmodeller: A JAX-based Python package for thermochemical equilibrium of planetary interiors and atmospheres'
tags:
  - Python
  - JAX
  - planetary science
  - exoplanets
  - geochemistry
  - thermodynamics
  - chemical equilibrium
  - atmospheres
  - magma oceans
authors:
  - name: Dan J. Bower
    orcid: 0000-0002-0673-4860
    affiliation: "1"
  - name: Paolo A. Sossi
    orcid: 0000-0002-1462-1882
    affiliation: "1"
affiliations:
  - name: Institute of Geochemistry and Petrology, Department of Earth and Planetary Sciences, ETH Zurich, Clausiusstrasse 25, 8092 Zurich, Switzerland
    index: 1
    ror: 057rvn534
date: 18 October 2026
bibliography: paper.bib

# Optional fields for papers that are part of a joint submission.
# For example, submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#
# If you are not making a joint submission you should remove these lines.
#
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Throughout the formation and evolution of rocky and volatile-rich planets,
volatiles are exchanged between planetary reservoirs such as the atmosphere and
the mantle. Gas species react with one another, dissolve into molten interiors,
and may condense, and at high temperatures and pressures they behave
non-ideally. Modelling these processes requires a self-consistent treatment of
chemical equilibrium and mass balance. `Atmodeller` is a Python library built on
JAX [@jax2018github] that performs these calculations. It combines the
convenience of a high-level language with just-in-time compilation, automatic
vectorisation, and automatic differentiation. The library is modular: its
subpackages for solubility laws, equations of state, and thermodynamic data can
each be used on their own, without solving a full chemical system. Applications
include the magma ocean stage of young planets, volcanic outgassing on Earth and
other Solar System bodies, and interior--atmosphere coupling on exoplanets such
as super-Earths and sub-Neptunes [@Bower2025]. Because its core is a general
thermodynamic solver, `Atmodeller` can also be used for laboratory
geochemistry, for example to calculate the equilibrium gas composition in a
gas-mixing furnace.

# Statement of need

% The paper must have a clearly labeled “Statement of need” section that:

% Clearly states what problems the software is designed to solve
% Identifies who the target audience is
% Explains the software’s relation to other work in the field
% This section should make a compelling case for why the software is needed and
% what gap it fills in the research ecosystem.

% Good: Clear problem statement, well-defined target audience, and context
% within the broader field
% OK: Addresses the key elements but could be more specific about audience or
% context
% Bad (not acceptable): Vague about the problem being solved, unclear target
% audience, or missing context

Calculating how volatiles partition between a planet's interior and atmosphere
requires solving gas-phase speciation, condensate stability, and volatile
solubility in melt together, often at pressures where gases are far from ideal.
Existing equilibrium codes typically address only part of this problem (see
State of the Field). `Atmodeller` solves it within one framework and is designed
for researchers in planetary science, exoplanet science, and geochemistry who
need such calculations, either standalone or as one component of a larger model.
Its class-based interface, built with Equinox [@kidger2021equinox], lets users
assemble a model from the species, solubility laws, and equations of state they
require. A model can represent a particular planet, specified by its mass and
radius, or be solved per kilogram of material when no specific planet is in
mind. Because these components are interchangeable and the reaction network is
derived automatically from the chosen species, a hierarchy of models of
increasing complexity can be built with the same code. This modularity suits
both rapid exploration and coupling to other numerical frameworks, which
commonly use Python as the interface between components. The high-level
interface and example notebooks also make `Atmodeller` suitable for teaching.

Thermodynamic data, solubility laws, and equations of state for planetary
conditions are often uncertain, calibrated over limited ranges, and revised as
new experiments become available. Exploring how the choice of model affects the
results therefore requires a transparent and flexible framework, in which users
can select between alternative models and modify or add their own. In
`Atmodeller`, each model is documented with its original source and calibration
range, and new models can be added by implementing a short protocol, often a
single method.

Because `Atmodeller` is built on JAX, the equilibrium solution, such as the
number of moles of each species, can be differentiated automatically with
respect to the input parameters in both forward and reverse mode. The
derivatives are obtained from the implicit function theorem at the converged
solution, so they are exact and independent of the iterations the solver took
to find it. Forward mode efficiently gives the full Jacobian with respect to a
few inputs, which can be supplied to an outer solver to help solve coupled or
nested models quickly and robustly. Reverse mode gives the gradient of a scalar
objective, such as a likelihood, with respect to many inputs at a cost
comparable to a single solve.

Just-in-time compilation makes `Atmodeller` fast enough to embed within other
frameworks, and automatic vectorisation solves a batch of conditions in a single
call. Without vectorisation, even a fast solver is often bottlenecked by the
outer Python loops that call it, which erode the performance it was designed to
deliver. Speed alone is therefore not enough: the full workflow, not only the
solver, must support vectorisation and differentiation, which is why
`Atmodeller` expresses the entire solve in JAX. Because JAX provides these
capabilities as transformations of ordinary Python functions, the code contains
little numerical boilerplate and consists mostly of the physics and chemistry,
which keeps it easy to inspect, verify, and extend. Together, these capabilities
make parameter sweeps, Monte Carlo sampling, and Bayesian parameter estimation
practical, including with gradient-based samplers such as Hamiltonian Monte
Carlo.

# State of the Field
% Authors must describe how their software compares to other commonly-used
% packages in the research area. When related tools exist, authors must provide
% a clear “build vs. contribute” justification explaining:

% Their unique scholarly contribution
% Why existing alternatives are insufficient for their research needs
% What gap their software fills
% Good: Thorough comparison with existing tools and clear explanation of why a
% new tool was necessary
% OK: Identifies related work and explains key differences
% Bad (not acceptable): Ignores existing similar tools or fails to justify why
% contributing to existing projects wasn’t appropriate

Chemical equilibrium codes used in planetary science generally originate from
one of two communities, astrophysics or the Earth sciences, and their design
reflects the priorities of each. Codes from astrophysics are built around
gas-phase thermochemistry for the low-pressure environments of stellar and
planetary atmospheres, and therefore typically assume ideal-gas behaviour. Codes
from the Earth sciences instead emphasise condensed phases and solutions, and
describe fluid behaviour at the high pressures and temperatures of planetary
interiors. `Atmodeller` bridges these two traditions, and we compare it with
representative codes from each below. Rather than survey the full capabilities
of each code, we focus on the features most relevant to `Atmodeller`:

- `CEA` [@Gordon1994] minimises the Gibbs energy of multiphase systems and
  remains widely used, including for lava-world atmospheres [@Charnoz2023],
  while Python codes such as `TEA` [@Blecic2016] apply the same approach to the
  gas phase alone. These codes assume ideal-gas behaviour, treat condensates as
  pure phases, and do not include volatile dissolution in melt.
- `GGchem` [@Woitke2018] provides equilibrium constants for around 550 molecules
  and Gibbs free energies for about 250 condensates, including 38 liquids, valid
  down to 100 K. It assumes ideal-gas behaviour, does not include volatile
  dissolution in melt, and selects the stable condensate set using
  supersaturation monitoring with hand-crafted criteria, which requires
  continuation from high to low temperature for robust convergence.
- `FastChem Cond` [@Kitzmann2024] extends the gas-phase chemistry of `FastChem`
  with about 290 liquid and solid condensates, and determines condensate
  stability using the approach of @Leal2016. Candidate condensates are
  preselected as those with activities above unity in the gas-phase solution,
  and the gas-phase and condensate systems are solved alternately until they
  converge, rather than as a single system. It assumes ideal-gas behaviour,
  considers only pure condensates, and does not include volatile dissolution in
  melt.
- `MAGMA` [@Fegley1987; @Schaefer2004] has long been the reference code for
  computing the vapour above silicate melts, using an ideal mixing of complex
  components model for melt activities and solving for the oxygen fugacity by
  mass balance. It has never undergone an official open-source release.
- `VapoRock` [@Wolf2023] computes the partial pressures of 34 gas species in the
  volatile-free Si–Mg–Fe–Al–Ca–Na–K–Ti–Cr–O system above a silicate melt, using
  the MELTS liquid model to provide component activities. It operates in the
  vapour-limited regime: the melt composition and oxygen fugacity are
  prescribed, and each species' partial pressure follows independently from mass
  action, so there is no mass balance between melt and gas. It assumes ideal-gas
  behaviour and does not include volatile elements such as H and C.
- `LavAtmos` [@vanBuchem2023] adopts the same MELTS-based melt activities as
  `VapoRock` to compute the vapour above a volatile-free melt, but instead of
  prescribing the oxygen fugacity it solves for it by assuming congruent
  vaporisation, such that the vapour has the same oxygen-to-cation ratio as the
  melt. `LavAtmos` 2.0 [@vanBuchem2025] extends this approach to
  volatile-bearing atmospheres by coupling the vaporisation reactions to
  `FastChem` gas-phase chemistry for C, H, N, S, and P, using nested
  `scipy.optimize.fsolve` iterations to reconcile the oxygen partial pressures
  of the melt and atmosphere. It assumes ideal-gas behaviour, does not
  include volatile dissolution in melt or condensation, and assumes that all
  atmospheric oxygen derives from vaporisation of the melt.
- `EVo` [@Liggins2022] models volcanic degassing in the C–O–H–S–N system,
  coupling gas speciation to melt solubility and redox along decompression
  paths. Its gas species are fixed by the choice of chemical system and the
  governing equations are reduced algebraically by hand, so adding species
  requires modifying the code. Real-gas behaviour is included through
  pure-species fugacity coefficients combined by ideal mixing, condensed phases
  are limited to graphite, and the equations are solved with
  `scipy.optimize.fsolve`.
- The global chemical equilibrium model [@Schlichting2022; @Young2023;
  @Werlen2025] couples the atmosphere, silicate melt, and metal of super-Earths
  and sub-Neptunes. It considers planetary interfaces at different temperatures,
  so is not strictly a pure equilibrium model but rather a coupled
  thermochemistry model. Separate versions of the code have been developed for
  different chemical systems, each defining its own species and reactions. Gases and 
  the silicate melt are treated as ideal, while non-ideal mixing is included for 
  light elements in the metal. Solutions are obtained by minimising a residual 
  cost function with gradient-based optimisers from an ensemble of initial guesses
  [@Grimm2026].
- `Reaktoro` [@Leal2015] is oriented towards aqueous geochemistry and does not
  provide solubility laws for volatiles in silicate melts, although these could
  be added. `Atmodeller` adopts the same strategy as `Reaktoro` for determining
  species stability [@Leal2016], in which every candidate species is included in
  the system and assigned a stability variable, removing the need for heuristic
  selection of the stable species.
- General-purpose frameworks such as `Cantera` [@Cantera] and `GEMS`
  [@Kulik2013] provide comprehensive multiphase equilibrium solvers with
  extensive non-ideal models, including real-gas equations of state. However,
  they are not built around the empirical solubility laws commonly used for
  volatiles in planetary melts, nor around planetary constraints such as a
  surface pressure set by the atmospheric mass. They also solve each condition
  separately, without vectorisation over conditions or derivatives of the
  equilibrium solution. Conversely, melt solubility models such as `VESIcal`
  [@Iacovino2021; @Wieser2022] compute volatile saturation and degassing in 
  silicate melts but do not solve for gas-phase speciation. Commercial packages 
  such as `FactSage` and `Thermo-Calc` offer extensive thermodynamic databases, 
  but are closed-source and require a paid licence.
- Many other models embed simple chemistry, often involving only a handful of
  species, within a broader and frequently time-dependent simulation. In these
  models the chemistry is typically tightly coupled to the host code, with
  hard-coded species, thermodynamic data, and solubility laws. We do not
  consider them here because they are not standalone chemistry packages, are not
  always publicly available, and cannot readily be extended, modified, or reused
  independently of the host model.

No existing open-source code combines real-gas equations of state, volatile
solubility in silicate melt, and coupling to solid and metal reservoirs with
automatic determination of species stability. `Atmodeller` is designed to fill
this gap, and many of its equations of state and solubility laws are tested
against published data.

The codes above also lack the differentiability and vectorisation that
`Atmodeller` provides. Jacobians are commonly constructed internally for use by
the solver, and `Reaktoro` uses automatic differentiation internally to provide
equilibrium sensitivities [@Leal2015], but no code exposes derivatives of its
solutions in a form that can be composed with other models. Chemistry therefore
cannot be coupled to other processes within a single differentiable model.
Likewise, none solves batches of conditions in a single vectorised call, which
JAX provides out of the box and which makes large parameter sweeps and
gradient-based inference practical.

Extending an existing code was considered but is not practical for these goals.
Differentiable, batched solves require every component, from thermodynamic data
and equations of state to solubility laws and the nonlinear solver, to be
expressed in a single array-programming framework. Retrofitting this into codes
written in C++ or Fortran, or built around `scipy.optimize`, would amount to
rewriting their core. Such codes can be called from JAX as external functions,
but derivatives and batching must then be supplied by hand for each one, and are
limited to what the external code exposes. Adding real-gas mixing, volatile
solubility, and coupling to interior reservoirs would likewise require
substantial changes to codes designed around ideal-gas atmospheres. The codes
above also differ in how actively they are developed: some are no longer
maintained, some receive occasional updates that are mostly bug fixes rather
than new features, and others are under active development. `Atmodeller` was
therefore built natively in JAX, with a modular design so that thermodynamic
data and models developed in both communities can be incorporated as components.

The standard library of thermodynamic data in `Atmodeller` focuses on the
species most relevant to planetary interiors and atmospheres, and is currently
smaller than those of astrophysical codes. Additional gas species and
condensates can be added straightforwardly from standard thermochemical tables,
and further non-ideal activity models for condensed phases can be added through
the same protocols as other components. Because `Atmodeller` is written natively
in JAX, it is most naturally called from Python, and the full benefit of
end-to-end differentiability is realised when the host model is also written in
JAX.

# New features in version 2.0.0

`Atmodeller` v2.0.0 extends the capabilities of v1.0.0 [@Bower2025] in several
ways:

1. **Thermodynamic phases.** The gas, silicate melt, silicate solid, metal, and
   pure condensed phases are represented explicitly, and any combination can be
   included. Species within a condensed phase form a solution, so mole fractions
   and activities are computed self-consistently, with optional
   activity-coefficient models such as that for Si in liquid Fe [@Badro2015]. A
   phase can include a background mass, such as a silicate mantle or an iron
   core, into which tracked species dissolve but which does not itself react.
2. **Solubility laws.** New laws include the empirical model of @MVC98 for
   H$_2$O in silicate melts, calibrated to 3 kbar.
3. **Non-ideal gas mixing.** Real-gas equations of state can depend on the gas
   composition, for example the mixture model of @ZD09 with Lorentz--Berthelot
   mixing rules, rather than treating each species as a pure gas.
4. **Pressure-dependent condensed phases.** The Gibbs energy of a condensed
   species can include a pressure correction from a volume model [@HP98; @HP11].
   This matters at the GPa pressures beneath thick envelopes, for example for
   the stability of graphite relative to diamond.
5. **A modular thermodynamic database.** Thermodynamic data are accessed through
   a database interface that users can extend with their own data. New species
   include graphite and diamond [@Gustafson1986], carbonates,
   Fe$_2$O$_3$, and deuterated species.
6. **Planetary structure.** The surface pressure can be calculated from a
   scaling law for planets with thick envelopes [@Schlichting2022], in addition
   to the hydrostatic balance for a thin atmosphere, or set directly for
   laboratory conditions. Other scaling laws can be added by subclassing a
   planet class and defining its surface pressure.
7. **End-to-end compilation and differentiation.** The whole workflow, from
   initial guess to solution, can be JIT-compiled, with the root finding
   performed by Optimistix [@optimistix2024]. Batches of conditions are solved
   in a single vectorised call, and an automatic retry strategy improves
   robustness across wide parameter ranges. Derivatives of the equilibrium
   solution can be computed in both forward and reverse mode, by implicit
   differentiation at the converged root, so they do not depend on the history
   of the solve. Model parameters, such as the coefficients of solubility laws
   and activity models, can also be differentiated.
8. **Initial guess.** A new initialiser estimates the element potentials (the
   Lagrange multipliers of the element mass balances) [@Gordon1994] for an ideal
   gas. Each element's mass balance is first solved separately, sweeping over
   the elements in order of decreasing abundance as in FastChem [@Stock2018]. A
   damped Newton method then solves the coupled problem, including the pressure
   constraint, dissolution in the melt, and any stable condensates. Condensates
   are switched on or off according to their stability, and the two steps are
   repeated for a fixed number of passes. The sweep keeps minor species in the
   solution when one species dominates. Although computing the guess takes time,
   the total run time is lower, because fewer iterations of the full solver are
   needed and the retry strategy is triggered less often. Alternatively, a
   previous solution can be supplied as a warm start, for example when coupling
   to a time-stepping code.
9. **Performance.** Species-specific models, such as equations of state,
   activity models, and solubility laws, are evaluated directly for each
   species, so only the models that are needed are evaluated, even under
   vectorisation. Solver modes that trade robustness features for shorter
   compilation are also provided, and state updates are applied in a single
   pass, which reduces overhead when `Atmodeller` is called repeatedly from
   another code.
10. **Outputs.** All outputs can be computed within JIT-compiled code, so
    `Atmodeller` can be embedded in other frameworks without loss of
    performance. New outputs include quantities of astronomical interest, such
    as elemental abundances relative to solar and the metallicity of the gas.
11. **Licence.** `Atmodeller` is now released under the permissive Apache 2.0
    licence, replacing GPL-3.0, so that it can be incorporated into other
    software, including proprietary and differently licensed codes, without
    imposing copyleft obligations.

Version 2.0.0 also substantially extends the testing of individual components.
Equations of state, solubility laws, activity models, and thermodynamic data are
tested against values from the original publications, such as tabulated
experimental data, published figures, and reference phase boundaries. In
addition, the codebase was systematically reviewed with AI coding agents under
the direction of the authors, both to probe the robustness of the solver and to
check model coefficients against the original publications. All resulting
changes were reviewed by the authors and verified against a comprehensive test
suite.

# Software design

% The paper must include a section explaining the architectural choices made:

% Trade-offs considered during design
% The design/architecture chosen and why
% Why these choices matter for the research application
% The content should be compelling and demonstrate meaningful design thinking,
% not just a superficial description of the code structure.

% Good: Thoughtful discussion of design decisions and their implications
% OK: Clear explanation of architecture with some rationale
% Bad (not acceptable): Generic code structure description without explaining
% design reasoning

`Atmodeller` uses JAX to combine a familiar high-level language with optimised
compiled code: models are written in Python, and JAX traces and compiles them
with XLA. JAX was designed around automatic differentiation: gradients are a
core transformation of the library, composable with compilation and
vectorisation, rather than an add-on supplied by separate tools as in many other
languages. Because JAX follows a functional programming paradigm, `Atmodeller`
uses `Equinox` [@kidger2021equinox] to provide a class-based design in which
physical entities, such as planets, phases, species, solubility laws, and
equations of state, are encapsulated as immutable modules that JAX can transform
directly. The physics and chemistry therefore remain readable and easy to
modify, while still benefiting from just-in-time compilation, vectorisation, and
automatic differentiation. Because it is written entirely in Python with no
compiled extensions of its own, `Atmodeller` installs with `pip` and runs on
Linux, macOS, and Windows. The main design challenge is preserving
differentiability throughout the model without sacrificing this readability.
This requires the usual care of JAX code, namely distinguishing static
configuration from traced quantities and avoiding branching logic that breaks or
obscures gradients.

The code separates the physical and chemical components from the numerical
engine that solves for equilibrium. Components interact with the engine only
through small protocols; for example, a solubility law returns a dissolved
concentration and an equation of state returns an activity. Components can
therefore be added or replaced without changing the solver. The engine assembles
the reaction network and mass balance from the chosen components into a system
of nonlinear equations, which is solved with the root finders of Optimistix
[@optimistix2024], together with an element-potential initial guess and an
automatic retry strategy for difficult cases.

A challenge common to solving for chemical equilibrium is that the solution need
not be unique. For a fixed set of species with ideal gas and ideal solution
behaviour at a single temperature and pressure, the Gibbs energy is convex and
the equilibrium is unique. In practice, however, empirical solubility laws and
activity models drawn from different sources are not derived from a single Gibbs
energy function, non-ideal mixing can introduce non-convexity, and computing the
surface pressure from the atmospheric mass couples the pressure to the solution
itself. Each of these can give rise to multiple solutions, or to none within the
physical domain, and a solver may converge to a metastable or non-physical
state.

`Atmodeller` does not eliminate this possibility, but mitigates it: unknowns are
solved in logarithmic form so that species amounts remain positive, convergence
is assessed directly from the residual of the governing equations rather than
from the solver's internal status, failed solves are retried from initial
guesses perturbed around converged neighbouring solutions, and systems with
species stability are solved by continuation in the stability variables. Because
solves are batched, exploring the sensitivity of a solution to its initial guess
is also inexpensive.

General-purpose thermodynamic packages typically require the pressure,
temperature, and bulk composition to be specified directly. `Atmodeller` instead
includes a lightweight physics layer that lets users specify inputs in the form
most natural to their problem, such as the mass and radius of a planet, its
surface temperature, elemental abundances by mass or by moles, and the
fugacities or activities of individual species, including oxygen fugacity
relative to a mineral redox buffer. The pressure can be fixed or determined
self-consistently, for example from the mass of the atmosphere through
hydrostatic balance or a scaling law. This makes the software much easier to use
for its target audience, particularly in planetary science and astronomy, where
detailed chemistry is not part of most researchers' daily work and the many
options of a general-purpose solver can be a barrier to entry. The physics layer
converts these inputs into a scaled chemical system for the solver and converts
the solution back into physical quantities for the user.

Accessibility also depends on users being able to trust the physics and
chemistry behind a result. Each component, such as an equation of state or a
solubility law, can therefore be isolated and evaluated on its own, outside the
equilibrium solve, and is linked in its documentation to the publication and
dataset on which it was calibrated. Substantial effort was made to keep each
implementation close to its original source, with tests that reproduce published
values where available. A result can therefore be traced back to the specific
models and data that produced it, which is difficult when physics and chemistry
are hard-coded deep within a code base with minimal documentation or comparison
to the original sources. This transparency helps the community connect the code
to the science it supports and builds confidence in its results.

# Research impact statement

% The paper must provide evidence of either:

% Realized impact:

% Publications using the software
% Evidence of external use and adoption
% Integration with other research tools or workflows
% OR credible near-term significance:

% Benchmark results demonstrating improvements
% Reproducible research materials showing capabilities
% Community-readiness signals (e.g., requests from other groups, presentations
% at relevant venues)
% The evidence should be compelling and specific, not aspirational.

% Good: Multiple citations, documented external users, or strong benchmark
% results with reproducible materials
% OK: Some evidence of use beyond the authors’ group or solid benchmarks
% Bad (not acceptable): Only statements about potential future use without
% concrete evidence

`Atmodeller` has already been used to study the atmospheres of rocky planets
[@Bower2025; @Seidler2026] and sub-Neptunes [@Bower2025; @Hakim2026], to
post-process the outcomes of planet formation models [@Polman2026], and has been
coupled to time-dependent models to explore the consequences of atmospheric loss
on rocky planets [@Cherubim2025]. Several of these studies were led by groups
outside the development team [@Cherubim2025; @Polman2026], and `Atmodeller` has
been adopted independently as an optional chemical-equilibrium backend in the 
JAX-based retrieval code ExoSkyrer [@Lee2026]. It has also been adopted as a 
chemistry module in `PROTEUS` [@Lichtenberg2021; @Nicholls2024; @Nicholls2025], a
community-developed framework for modelling the coupled evolution of planetary
interiors and atmospheres. `Atmodeller` grew out of `CALLIOPE`, a prototype
volatile module previously used in `PROTEUS`.

Beyond these applications, `Atmodeller` brings to planetary science the
vectorised and differentiable modelling that JAX has made practical in other
fields. Most existing planetary models are built on established codes that lack
these capabilities, so coupling chemistry to other processes within a single
differentiable model has not been practical. `Atmodeller` provides a chemistry
component that such models can adopt directly, and its design offers a template
for new codes, supporting the move towards a new generation of differentiable
models in Earth and planetary sciences.

% [TODO: benchmark, e.g. "A batch of N conditions solves
% in X s on a single CPU/GPU, compared with Y s for a Python loop over the same
% solver", with a pointer to the reproducible notebook.]

%# Mathematics

%Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

%Double dollars make self-standing equations:

%$$\Theta(x) = \left\{\begin{array}{l}
%0\textrm{ if } x < 0\cr
%1\textrm{ else}
%\end{array}\right.$$

%You can also use plain \LaTeX for equations
%\begin{equation}\label{eq:fourier}
%\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
%\end{equation}
%and refer to \autoref{eq:fourier} from text.

% # Citations

% Citations to entries in paper.bib should be in
% [rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
% format.

% If you want to cite a software repository URL (e.g. something on GitHub without
% a preferred citation) then you can do it with the example BibTeX entry below for
% @fidgit.

% For a quick reference, the following citation commands can be used: -
% `@author:2001`  ->  "Author et al. (2001)" - `[@author:2001]` -> "(Author et
% al., 2001)" - `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001;
% Author2 et al., 2002)"

% # Figures

% Figures can be included like this:
% ![Caption for example figure.\label{fig:example}](figure.png)
% and referenced from text using \autoref{fig:example}.

% Figure sizes can be customized by adding an optional second parameter:
% ![Caption for example figure.](figure.png){ width=20% }

# AI usage disclosure

% The paper must include a section that transparently discloses any use of
% generative AI in:

% Software creation or development
% Documentation writing
% Paper authoring
% If no AI tools were used, this should be explicitly stated. If AI tools were
% used, authors should describe how they were used and how the quality and
% correctness of AI-generated content was verified.

% Good: Clear, specific disclosure of AI tool usage and verification methods
% OK: Statement that no AI tools were used, or general disclosure of AI
% assistance
% Bad (not acceptable): Missing section or vague/evasive language about AI usage

Version 1 of `Atmodeller` was developed with limited, targeted use of ChatGPT
(OpenAI; GPT-4 and GPT-5 models) for specific parts of the codebase. With that
foundation established, AI coding agents, principally Claude Code (Anthropic),
have become an integral part of developing version 2, used to implement new
features, refactor existing code, and review the codebase as described above.
Throughout, the authors directed this work, setting the design, scope, and
physical assumptions of each change and steering the AI output to keep the
codebase as accessible and understandable as possible to its users, who are
primarily scientists and researchers. The authors reviewed every line of
AI-generated code and verified all changes against the test suite before
accepting them. Claude Code was also used to edit the text of this paper, which
the authors reviewed in full.

# Acknowledgements

`Atmodeller` has benefitted from the work of many in the chemical equilibrium
community, including the authors of the studies and codes cited above, both for
elucidating methodology and numerical techniques and for making their codes
open source and available for benchmarking and code comparison. We also thank
the JAX community, and Patrick Kidger and Johanna Haffner in particular, for
their sustained efforts in developing and supporting the JAX ecosystem on which
`Atmodeller` is built.

# References