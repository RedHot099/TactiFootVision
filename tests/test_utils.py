"""``tv.utils`` helpers that the backends share."""

import sys

import pytest

from tactifoot_vision.utils import require_extra


def test_require_extra_returns_the_module_when_installed():
    assert require_extra("json", "nope").dumps({"a": 1}) == '{"a": 1}'


def test_require_extra_names_the_extra_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "rfdetr", None)
    with pytest.raises(
        ImportError,
        match=r'rfdetr is not installed: run `pip install "tactifoot-vision\[rfdetr\]"`',
    ):
        require_extra("rfdetr", "rfdetr")


def test_require_extra_takes_a_command_and_a_hint_instead(monkeypatch):
    monkeypatch.setitem(sys.modules, "trackeval", None)
    with pytest.raises(ImportError, match=r"run `pip install x`, or pass hota=False"):
        require_extra("trackeval", install="pip install x", hint=", or pass hota=False")


def test_require_extra_lets_a_broken_module_raise_its_own_error(tmp_path, monkeypatch):
    (tmp_path / "umap").mkdir()
    (tmp_path / "umap" / "__init__.py").write_text("import numba_that_is_missing\n")
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, "umap", raising=False)
    with pytest.raises(ModuleNotFoundError, match="numba_that_is_missing"):
        require_extra("umap", "teams")
