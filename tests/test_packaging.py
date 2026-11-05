"""The built distributions: what goes to PyPI and what stays in the working tree."""

import tarfile
from pathlib import Path

import pytest
from hatchling.build import build_sdist

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sdist_members(tmp_path_factory) -> list[tarfile.TarInfo]:
    """The members of a freshly built sdist, as PEP 517 would build it."""
    out = tmp_path_factory.mktemp("sdist")
    name = build_sdist(str(out), config_settings=None)  # reads pyproject.toml of cwd
    with tarfile.open(out / name) as tar:
        return tar.getmembers()


@pytest.fixture(autouse=True, scope="module")
def _from_the_repo_root():
    import os

    before = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(before)


def _paths(members: list[tarfile.TarInfo]) -> set[str]:
    # "tactifoot_vision-X.Y.Z/<path>" -> "<path>"
    return {m.name.split("/", 1)[1] for m in members if "/" in m.name}


def test_sdist_holds_the_package_and_its_papers(sdist_members):
    paths = _paths(sdist_members)
    assert "src/tactifoot_vision/__init__.py" in paths
    assert "src/tactifoot_vision/py.typed" in paths
    assert {"README.md", "LICENSE", "pyproject.toml", "PKG-INFO"} <= paths


def test_sdist_leaves_the_working_tree_behind(sdist_members):
    top = {p.split("/", 1)[0] for p in _paths(sdist_members)}
    # hatchling always adds a .gitignore; nothing else from the tree gets in
    assert top == {
        "src",
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "PKG-INFO",
        ".gitignore",
    }


def test_sdist_has_no_symlinks(sdist_members):
    links = [m.name for m in sdist_members if m.issym() or m.islnk()]
    assert links == []
