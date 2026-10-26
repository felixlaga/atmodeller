HYDROGEN STORAGE AND THE SILANE / METHANE DIAGNOSTIC

This directory contains a numerical investigation using the locally cloned
Atmodeller v2.0.0 at commit f1e3d94041ce6056a4995f7d55d11edeeaee5de1.
This is the historical investigation before the spinodal extension.
The stored outputs retain the original upstream commit provenance.

Question: Does changing interior hydrogen storage alter atmospheric SiH4/CH4
after cooling and equilibrium condensation?

This is a constitutive sensitivity study. Multipliers on published H2 and H2O
solubility laws are hypothetical scenarios, not a spinodal-decomposition model
or experimental confidence intervals. There is no nonconvex mixing free energy,
binodal/spinodal calculation, nucleation, diffusion, thermal evolution, rainout,
photochemistry, chemical quenching, or synthetic spectrum in these runs.

REPRODUCE FROM THE REPOSITORY ROOT

The historical calculations used Python 3.13 and the versions recorded in
requirements.lock.txt. The original engine commit is recorded above. For an
exact engine comparison, install that commit in a separate environment using:

  python3 -m venv .research-venv
  .research-venv/bin/python -m pip install -r research/hydrogen_storage/investigation/requirements.lock.txt "atmodeller @ git+https://github.com/ExPlanetology/atmodeller.git@f1e3d94041ce6056a4995f7d55d11edeeaee5de1"

To exercise the scripts against this fork instead, use its editable environment
(uv sync --locked --all-extras --dev), then run:

  .venv/bin/python research/hydrogen_storage/investigation/validate_reference.py
  .venv/bin/python research/hydrogen_storage/investigation/validate_atmosphere.py
  .venv/bin/python research/hydrogen_storage/investigation/run_study.py
  .venv/bin/python research/hydrogen_storage/investigation/run_atmospheres.py
  .venv/bin/python research/hydrogen_storage/investigation/check_numerics.py
  .venv/bin/python research/hydrogen_storage/investigation/analyse.py
  .venv/bin/python research/hydrogen_storage/investigation/build_report.py

The stored report and numerical arrays are historical artifacts. Its original
reproduction paths refer to the earlier standalone workspace; use these commands
after migration. build_report.py rebuilds the report from stored data and keeps
their engine commit; it does not prove that the data were rerun with a newer commit.
Record a new engine revision if replacing the archived results with new runs.

The JAX compilation cache stays in outputs/.jax_cache. First compilation and
continuation through stiff equilibria can take several minutes. run_study.py
--resume resumes its own recorded checkpoint. Fresh runs do not use that flag.

FILES

model.py: precise chemical network, inventories, hypothetical scaling laws,
and extraction of phase-resolved conserved-element budgets.

validate_reference.py: 100 deep-boundary regressions against the authors'
1x- and 100x-solar workbooks. Initial guesses come from the archive, but all
reported values are re-solved with the pinned current Atmodeller source.

validate_atmosphere.py: 50 atmospheric regressions using the archive's actual
elemental inventories and pressures. This isolates atmospheric chemistry from
small differences between code versions in the deep-boundary calculation.

run_study.py: fixed total elemental budgets; H2, H2O and joint solubility
perturbations; pressure-feedback controls; temperature and metallicity checks.
Nonconvergent cases are excluded and explicitly recorded in study_meb.json.
A failed solve is not evidence for a physical instability.

run_atmospheres.py: local, imposed-temperature/pressure equilibrium parcels
normalized to one kg of the computed MEB gas inventory. Condensates stay in
equilibrium with the gas. Separate runs disable condensates as a control.

analyse.py: plots and direct Gibbs-energy checks of Si+SiC and C+SiC buffers.

sources/: unmodified authors' files from https://osf.io/ac6bw/ plus the OSF API
manifest and SHA256 provenance. The extracted notebook is for reference and
must not be executed as a current-version reproduction script.

outputs/: numerical arrays, JSON results, validation, exclusions and provenance.
figures/: PNG, SVG and PDF scientific plots.
report.html: detailed results and interpretation.

Model conventions: masses kg; temperatures K; pressures and fugacities bar;
natural logarithms internally; reported gas abundances are mole fractions.
Atmodeller names SiH4 as H4Si, SiO as OSi, NH3 as H3N, SiO2 as O2Si and
Si3N4 as N4Si3. The _g, _d, _s and _l suffixes denote phase/state.

The archived solvent-mass convention is retained: tracked dissolved volatile
mass is conserved but does not add to the fixed background melt mass used in
the dissolution laws and gravity. The archive's core mass fraction is
0.295334691460966; the silicon inventory calculation separately uses the
notebook's 0.327 when deriving its rounded 0.145 planetary Si mass fraction.

Exploratory diagnostic scripts and logs record numerical troubleshooting only;
they do not supply accepted scientific results. Only converged, conserved
results in the main study/atmosphere JSON files feed the report.
