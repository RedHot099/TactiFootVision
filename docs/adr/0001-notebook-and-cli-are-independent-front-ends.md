# Notebook and CLI are independent front-ends over one core

The package has two modes: Python imports in a notebook, and the `tactifoot` command. Both call the same package functions and classes. The CLI only translates command-line text into those calls. It has no logic of its own and declares no default values.

Concretely:

- `tactifoot train` generates its flags from `TrainConfig`.
- Each section of a run file maps onto one constructor or function, and its keys are checked against that signature.
- Only the constructors and `TrainConfig` hold default values.

As a result, a change in the core reaches both modes without editing `cli.py`.

## Considered options

- **One shared run-config object** that both modes go through: load YAML, edit it in Python, save it back. Rejected because the notebook should stay plain Python, with no config object between the user and the classes.
- **Hand-written CLI flags and config sections that restate defaults.** Rejected because core changes silently skipped the CLI. A new `TrainConfig` field had no flag. A changed `HomographyEstimator` default did not reach `tactifoot run`, since the config passed its own copy explicitly.
