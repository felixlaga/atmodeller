.. _DevelopersGuideFile:

Developer's Guide
=================

Introduction
------------

Community development of *Atmodeller* is strongly encouraged, and bug fixes and new features are welcome. *Atmodeller* uses JAX so familiarise yourself with `How to think in JAX <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html>`_ and `JAX - The Sharp Bits <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_, as well as other resources offered on the JAX site and the web in general. You are welcome to enquire about the reasoning behind the structure and design of the code with the development team.

Contributing
------------

Contributions follow the usual GitHub workflow:

1. For larger changes, open an `issue <https://github.com/ExPlanetology/atmodeller/issues>`__ first so the approach can be discussed.
2. Fork the repository to your own account (personal or organisational).
3. Make your changes on a branch, including tests and documentation where required (see below).
4. Run the tests and pre-commit, then open a pull request against the upstream repository and complete the pull request template.

To report a bug, request a feature or ask for help, please open an `issue <https://github.com/ExPlanetology/atmodeller/issues>`__. For bugs, include a minimal example that reproduces the problem, the versions of *Atmodeller* and JAX you are using, and any error message.

Under the Apache License 2.0 you are not required to make your changes public, but we encourage you to contribute them back so the whole community benefits. Contributions submitted by pull request are licensed under the same terms as the project.

Installation
------------

See :ref:`developer_install` for how to install *Atmodeller* and the extra dependencies required for the developer tools and documentation.
 
Pre-commit
----------

Run pre-commit before issuing a pull request to the main repository::

    pre-commit run --all-files

Documentation
-------------

.. note::
    Documentation should be consistent with the source code, so any changes to the source code should be reflected in changes to the documentation, if required.

The doc dependencies include a bundled copy of pandoc (via ``pypandoc-binary``), which is used if pandoc is not already installed. On Apple Silicon Macs the bundled copy is currently an x86_64 binary that requires Rosetta, so either install pandoc natively (``brew install pandoc``) or install Rosetta (``softwareupdate --install-rosetta``). When the doc dependencies have been installed the documentation can be compiled::

    cd docs
    sphinx-apidoc -f -o source ../atmodeller

To generate HTML documentation use the appropriate command for your operating system:

Mac/Linux::

    make html

Windows::

    .\make html

To generate PDF documentation, noting that a LaTeX distribution must be available on your system::

    make latexpdf

Documentation is built in the appropriately named subdirectory in ``_build``.

Tests
-----

.. note::
    Tests should be consistent with the source code, so any changes to the source code should be reflected in changes to the test suite, if required.

You can confirm that all tests pass by navigating to the root directory of *Atmodeller* and running::
    
    pytest

The performance tests, which time the solver for batches of models, are slow and are skipped by default. To run them::

    pytest -m performance
    
Similarly, to test coverage::

    pytest --cov

The percentage test coverage is reported in ``README.md`` but requires manual update.

Add a corresponding test for new features that are developed.