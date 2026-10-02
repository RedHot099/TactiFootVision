"""Run files: YAML that says how ``tactifoot run`` processes video.

Each section maps onto one package constructor or function, and its keys are
that callable's keyword arguments::

    detector: {type: yolo, weights: ../models/football_yolo11m.pt, conf: 0.3}
    keypoints: {type: yolo_pose, weights: ../models/pitch_yolov8n_pose.pt}
    tracker: {type: bytetrack, lost_track_buffer: 30}
    teams: {embedder: siglip}
    render: {annotator: {style: video_game}}

==============  ==========================================================
``detector``    ``tv.models.load_model(type, **rest)`` (required)
``keypoints``   same; absent or ``null``: no keypoint model
``tracker``     ``tv.tracking.create_tracker(type, **rest)``; ``null``: no tracking
``teams``       ``TeamClassifier(**section)``; absent or ``null``: no teams
``pitch``       ``SoccerPitch(**section)``
``homography``  ``HomographyEstimator(pitch, **section)``
``pipeline``    the remaining ``Pipeline`` keyword arguments
``render``      ``annotator`` (``FrameAnnotator``), ``radar`` (``PitchRadar``;
                ``null``: none) and ``render_video`` keyword arguments
==============  ==========================================================

Keys are checked against the signatures when the file is loaded; values are
checked by the constructors when :meth:`RunFile.build_pipeline` runs. A key left
out keeps the package default, so this module holds no defaults of its own.
String values starting with ``./`` or ``../`` are paths relative to the file.
"""

import copy
import inspect
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tactifoot_vision.registry import Registry

if TYPE_CHECKING:
    from tactifoot_vision.pipeline import Pipeline, PipelineResult

SECTIONS = (
    "detector",
    "keypoints",
    "tracker",
    "teams",
    "pitch",
    "homography",
    "pipeline",
    "render",
)
# Pipeline arguments that other sections build.
_PIPELINE_PARTS = (
    "detector",
    "keypoint_model",
    "tracker",
    "team_classifier",
    "pitch",
    "homography",
)
# render_video arguments the run file does not set.
_RENDER_ARGUMENTS = ("result", "source", "output", "annotator", "radar", "progress")
_NULLABLE = {"keypoints", "tracker", "teams"}


def load(path: str | Path, overrides: Iterable[str] = ()) -> "RunFile":
    """Read a run file, apply ``key.path=value`` overrides and check every key.

    Raises ``ValueError`` naming the section and the valid keys when a key is
    unknown. Override values are YAML (``null``, ``true``, ``0.3``, ``[a, b]``);
    paths in them starting with ``./`` or ``../`` are relative to the current
    working directory.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    data = _resolve_paths(data, path.parent.absolute())
    for override in overrides:
        key, value = parse_override(override)
        _set(data, key, value)
    _check(data)
    return RunFile(path, data)


def parse_override(text: str) -> tuple[str, Any]:
    """Split ``"dotted.key=value"`` into the key and its YAML-parsed value.

    Paths in the value starting with ``./`` or ``../`` are made absolute
    against the current working directory.
    """
    key, sep, value = text.partition("=")
    if not sep:
        raise ValueError(f"Override {text!r} needs the form key=value")
    if not key or any(not part for part in key.split(".")):
        raise ValueError(f"Override {text!r} has an empty key")
    return key, _resolve_paths(yaml.safe_load(value), Path.cwd())


class RunFile:
    """A loaded, key-checked run file (see :func:`load`)."""

    def __init__(self, path: Path, sections: dict[str, Any]) -> None:
        self.path = path
        self._sections = sections

    @property
    def sections(self) -> dict[str, Any]:
        """A copy of the sections after overrides and path resolution."""
        return copy.deepcopy(self._sections)

    def build_pipeline(self) -> "Pipeline":
        """Create the models, tracker, team classifier and :class:`Pipeline`."""
        from tactifoot_vision.models import load_model
        from tactifoot_vision.pipeline import Pipeline
        from tactifoot_vision.pitch import HomographyEstimator, SoccerPitch
        from tactifoot_vision.teams import TeamClassifier
        from tactifoot_vision.tracking import create_tracker

        sections = self.sections
        kwargs = sections.get("pipeline", {})
        kwargs["detector"] = _create(load_model, sections["detector"])
        if sections.get("keypoints") is not None:
            kwargs["keypoint_model"] = _create(load_model, sections["keypoints"])
        if "tracker" in sections:
            tracker = sections["tracker"]
            kwargs["tracker"] = (
                None if tracker is None else _create(create_tracker, tracker)
            )
        if sections.get("teams") is not None:
            kwargs["team_classifier"] = TeamClassifier(**sections["teams"])
        if "pitch" in sections:
            kwargs["pitch"] = SoccerPitch(**sections["pitch"])
        if "homography" in sections:
            pitch = [kwargs["pitch"]] if "pitch" in kwargs else []
            kwargs["homography"] = HomographyEstimator(*pitch, **sections["homography"])
        return Pipeline(**kwargs)

    def render_video(
        self,
        result: "PipelineResult",
        source: str | Path | None,
        output: str | Path,
        progress: bool = True,
    ) -> Path:
        """:func:`tactifoot_vision.viz.render_video` with the ``render`` section."""
        from tactifoot_vision.viz import FrameAnnotator, PitchRadar, render_video

        kwargs = self.sections.get("render", {})
        if "annotator" in kwargs:
            kwargs["annotator"] = FrameAnnotator(
                pitch=result.pitch, **kwargs["annotator"]
            )
        if "radar" in kwargs:
            radar = kwargs["radar"]
            kwargs["radar"] = (
                False if radar is None else PitchRadar(pitch=result.pitch, **radar)
            )
        return render_video(result, source, output, progress=progress, **kwargs)

    def __repr__(self) -> str:
        return f"RunFile({str(self.path)!r}, sections={list(self._sections)})"


def _create(factory: Callable[..., Any], section: dict[str, Any]) -> Any:
    """``load_model`` / ``create_tracker`` called with a ``{type: name, **options}`` section."""
    options = dict(section)
    return factory(options.pop("type"), **options)


# ------------------------------------------------------------------ checking
def _check(data: dict[str, Any]) -> None:
    from tactifoot_vision.models import MODELS
    from tactifoot_vision.pipeline import Pipeline
    from tactifoot_vision.pitch import HomographyEstimator, SoccerPitch
    from tactifoot_vision.teams import TeamClassifier
    from tactifoot_vision.tracking import TRACKERS

    unknown = [name for name in data if name not in SECTIONS]
    if unknown:
        raise ValueError(
            f"Unknown run file section {unknown[0]!r}; valid sections: {', '.join(SECTIONS)}"
        )
    if "detector" not in data:
        raise ValueError("The run file needs a 'detector' section")
    for name, section in data.items():
        if section is None and name in _NULLABLE:
            continue
        section = _mapping(name, section)
        if name in ("detector", "keypoints"):
            _check_typed(name, section, MODELS)
        elif name == "tracker":
            _check_typed(name, section, TRACKERS)
        elif name == "teams":
            _check_keys(name, section, TeamClassifier)
        elif name == "pitch":
            _check_keys(name, section, SoccerPitch)
        elif name == "homography":
            _check_keys(name, section, HomographyEstimator, exclude=("pitch",))
        elif name == "pipeline":
            _check_keys(name, section, Pipeline, exclude=_PIPELINE_PARTS)
        else:
            _check_render(name, section)


def _check_render(name: str, section: dict[str, Any]) -> None:
    from tactifoot_vision.viz import FrameAnnotator, PitchRadar, render_video

    if "annotator" in section:
        _check_keys(
            f"{name}.annotator",
            _mapping(f"{name}.annotator", section["annotator"]),
            FrameAnnotator,
            exclude=("pitch",),
        )
    if section.get("radar") is not None:
        _check_keys(
            f"{name}.radar",
            _mapping(f"{name}.radar", section["radar"]),
            PitchRadar,
            exclude=("pitch",),
        )
    _check_keys(
        name,
        section,
        render_video,
        exclude=_RENDER_ARGUMENTS,
        extra=("annotator", "radar"),
    )


def _check_typed(name: str, section: dict[str, Any], registry: Registry[Any]) -> None:
    kind = section.get("type")
    if not isinstance(kind, str):
        raise ValueError(
            f"Section {name!r} needs a 'type' key, one of: {', '.join(registry.names())}"
        )
    try:
        factory = registry.get(kind)
    except ValueError as exc:
        raise ValueError(f"Section {name!r}: {exc}") from None
    rest = {k: v for k, v in section.items() if k != "type"}
    _check_keys(name, rest, factory, extra=("type",))


def _check_keys(
    name: str,
    section: Mapping[str, Any],
    target: Callable[..., Any],
    exclude: Iterable[str] = (),
    extra: Iterable[str] = (),
) -> None:
    """Fail when ``section`` has a key ``target`` does not accept as a keyword."""
    parameters = inspect.signature(target).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return
    valid = [*extra] + [
        p.name
        for p in parameters
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and p.name not in exclude
    ]
    for key in section:
        if key not in valid:
            raise ValueError(
                f"Unknown key {key!r} in section {name!r}; valid keys: {', '.join(valid)}"
            )


def _mapping(name: str, section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        nullable = " or null" if name in _NULLABLE or name == "render.radar" else ""
        raise ValueError(
            f"Section {name!r} must be a mapping{nullable}, got {section!r}"
        )
    return section


# ------------------------------------------------------------ paths, overrides
def _resolve_paths(value: Any, base: Path) -> Any:
    """Make strings starting with ``./`` or ``../`` absolute against ``base``, at any depth."""
    if isinstance(value, dict):
        return {key: _resolve_paths(item, base) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base) for item in value]
    if isinstance(value, str) and value.startswith(("./", "../")):
        return os.path.abspath(base / value)
    return value


def _set(data: dict[str, Any], key: str, value: Any) -> None:
    """Set ``data["a"]["b"] = value`` for ``key="a.b"``, creating missing mappings."""
    *parents, last = key.split(".")
    node = data
    for depth, part in enumerate(parents):
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            where = ".".join(parents[: depth + 1])
            raise ValueError(
                f"Cannot override {key!r}: {where!r} is {child!r}, not a mapping"
            )
        node = child
    node[last] = value
