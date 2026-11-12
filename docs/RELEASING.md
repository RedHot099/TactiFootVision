# Releasing

A release is a GitHub Release created by the maintainer. Nothing else creates
tags or publishes: agents never do either (global rule 3).

## Before the release

1. The GPU gate on omarchy is green on the commit to release: `tf-lab run
   model-tests`, `tf-lab run sam2` and `tf-lab run cli-e2e`. "Skipped" is not
   "passed".
2. CI is green on `master` (`ruff`, `fast suite`, `build and twine check`,
   `colab smoke`). The smoke installs the wheel with pip under
   `tests/colab-constraints.txt`; if Colab has moved (compare
   https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt
   with the file), update the constraints first and rerun the gate.
3. The maintainer opens the overview notebook from the README badge in Google
   Colab, on the current default runtime, installing the TestPyPI pre-release
   of the version about to ship, and runs it top to bottom. This is a manual
   step; no agent can do it. The notebook's first cell installs the latest
   PyPI release; for the pre-release, replace its `%pip` line with these two
   (the package from TestPyPI, its dependencies from PyPI) before running:

   ```
   %pip install -q --no-deps --index-url https://test.pypi.org/simple/ tactifoot-vision==X.Y.ZrcN
   %pip install -q tactifoot-vision==X.Y.ZrcN
   ```

   Use a GPU runtime and an `HF_TOKEN` secret whose account has accepted the
   conditions of the demo dataset.
4. Decide the version (0.x SemVer: minor for features and breaking changes,
   patch for fixes). The version is the tag; `pyproject.toml` holds none.

## The release

1. On GitHub, Releases, "Draft a new release": tag `vX.Y.Z` on `master`
   (created by the release), title `vX.Y.Z`, "Generate release notes" (from
   the pull request titles), "Set as a pre-release" for an `rcN`.
2. Publishing the release starts `.github/workflows/release.yml`: it builds,
   checks that the built version equals the tag, publishes to TestPyPI, then
   waits in the `pypi` environment.
3. Approve the `pypi` deployment on the workflow run page. The job publishes
   to PyPI. A pre-release stays on TestPyPI unless approved too.
4. Check https://pypi.org/project/tactifoot-vision/ renders the README, then
   `pip install tactifoot-vision==X.Y.Z` in a clean venv once.

## One-time setup (Trusted Publishing)

Both indexes trust this repository's `release.yml` through OIDC; no API token
is stored anywhere. `scripts/release-setup-wizard.sh` walks through the
clicks: a pending publisher on test.pypi.org and on pypi.org (owner
`RedHot099`, repository `TactiFootVision`, workflow `release.yml`,
environments `testpypi` and `pypi`), and the two GitHub environments, `pypi`
with the maintainer as required reviewer.

## Branch protection

`master` takes pull requests only, with the four CI jobs required, no force
pushes or deletions, zero required GitHub reviews (the reviewer agent reviews)
and administrators exempt. `scripts/release-setup-wizard.sh` applies it with
`gh api` once `ci.yml` is on `master`.
