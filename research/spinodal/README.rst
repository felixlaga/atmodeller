Binary-demixing implementation validation
========================================

``validation.json`` records the implementation commit, source hashes, locked
environment, local checks and physical scope. ``example_output.json`` is the
output of ``examples/spinodal_equilibrium.py``. Text logs retain the test,
style/type, documentation and changed-line coverage results. Local checkout
paths in logs have been replaced with ``<checkout>`` and trailing whitespace
has been removed.

The hydrogen-silicate example evaluates the published binary mixing function.
The gas-equilibrium example uses an illustrative, uncalibrated Fe-Si interaction
to verify integration with the native reaction engine. These are distinct
checks; neither establishes a calibrated sub-Neptune atmosphere or a
time-dependent decomposition model.

The implementation adds 36 tests; the full non-performance suite has 263 passing
tests. The three performance tests were intentionally deselected by the
repository's default configuration. Changed executable package lines have
100% line coverage. The validation records local CPU execution only.

The tests were followed by formatting-only changes to two test files and the
archive verification script; Python AST equality was checked. Package source
was unchanged between the full test run and the recorded implementation commit.
