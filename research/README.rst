Research accompanying the spinodal extension
===========================================

``hydrogen_storage/investigation/report.html`` contains the completed hydrogen
storage investigation, including its numerical results, plots, source workbooks,
validation, and reproduction scripts. See ``investigation/README.txt`` for
commands from the repository root. The earlier literature brief and upstream
installation check are also retained under ``hydrogen_storage/``.

These are historical results from unmodified Atmodeller commit
``f1e3d94041ce6056a4995f7d55d11edeeaee5de1``. The solubility multipliers in that
study are hypothetical sensitivity experiments, not spinodal physics. They
must not be relabelled as predictions of the new mixing model.

``hydrogen_storage/import_manifest.json`` records the original hashes and the
few path/provenance edits made during migration. Numerical outputs, scientific
figures and authors' downloads remain byte-identical. The authors' files retain
their original attribution and provenance in ``sources/downloaded.json`` and
the OSF manifest. Archive scripts are excluded from automatic style rewriting
to preserve this evidence; package code and new tests use the normal checks.
