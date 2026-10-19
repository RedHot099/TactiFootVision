"""The fixed vocabularies in ``tv.enums`` (ADR 0004)."""

import inspect
import pickle
import subprocess
import sys
from enum import StrEnum

import pytest

import tactifoot_vision as tv
from tactifoot_vision import enums
from tactifoot_vision.enums import Role, Split, Vocabulary

VOCABULARIES = [
    value
    for value in vars(enums).values()
    if inspect.isclass(value)
    and issubclass(value, Vocabulary)
    and value is not Vocabulary
]


def test_every_vocabulary_is_a_str_enum_whose_members_equal_their_values():
    assert len(VOCABULARIES) == 18
    for enum in VOCABULARIES:
        assert issubclass(enum, StrEnum) and enum.__doc__
        for member in enum:
            assert member == member.value and str(member) == member.value
            assert member.name == member.value.upper().replace("-", "_")


def test_a_member_prints_and_pickles_as_its_plain_string():
    assert repr(Role.GOALKEEPER) == "'goalkeeper'"
    assert f"{[Split.VALID]}" == "['valid']"
    loaded = pickle.loads(pickle.dumps({enums.DetectionKey.PITCH_XY: [Role.PLAYER]}))
    assert loaded == {"pitch_xy": ["player"]}
    assert type(next(iter(loaded))) is str and type(loaded["pitch_xy"][0]) is str
    assert b"tactifoot_vision" not in pickle.dumps(Role.PLAYER)


def test_parse_names_the_parameter_and_the_allowed_values():
    assert enums.Pooling.parse("cls", "pooling") is enums.Pooling.CLS
    assert enums.Pooling.parse(enums.Pooling.CLS) is enums.Pooling.CLS
    with pytest.raises(
        ValueError, match=r"Unknown pooling 'max'\. Available: mean, cls"
    ):
        enums.Pooling.parse("max", "pooling")
    with pytest.raises(ValueError, match="Unknown Reducer 'pca'. Available: umap"):
        enums.Reducer.parse("pca")


def test_split_parse_takes_the_aliases():
    assert Split.parse("val") is Split.parse("validation") is Split.VALID
    assert [Split.parse(s) for s in ("train", "valid", "test")] == list(Split)
    with pytest.raises(ValueError, match="Unknown split 'dev'. Available: train"):
        Split.parse("dev")


def test_a_class_name_outside_the_roles_is_other():
    assert Role.of("goalkeeper") is Role.GOALKEEPER
    assert Role.of("ball") is Role.of("sports ball") is Role.OTHER


def test_the_built_in_names_are_the_registries_names():
    # A fresh interpreter: test modules register plugins of their own.
    code = (
        "import tactifoot_vision as tv; from tactifoot_vision import enums; "
        "pairs = [(enums.ModelType, tv.models.MODELS), "
        "(enums.TrackerType, tv.tracking.TRACKERS), "
        "(enums.EmbedderType, tv.teams.EMBEDDERS), "
        "(enums.PostProcessName, tv.pipeline.POSTPROCESSES)]; "
        "print([(sorted(e), r.names()) for e, r in pairs "
        "if sorted(e) != r.names() or any(type(n) is not str for n in r.names())])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_task_and_truth_keep_their_old_import_paths():
    from tactifoot_vision.data.annotations import Task as AnnotationsTask
    from tactifoot_vision.data.labelled_clip import Truth as ClipTruth

    assert AnnotationsTask is tv.data.Task is enums.Task
    assert ClipTruth is tv.data.Truth is enums.Truth


def test_the_enums_module_imports_nothing_else_of_the_package():
    code = (
        "import sys, tactifoot_vision.enums; "
        "print(sorted(m for m in sys.modules if m.startswith('tactifoot_vision')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "['tactifoot_vision', 'tactifoot_vision.enums']"
