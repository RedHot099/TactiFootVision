"""Space control: for each point of the pitch, which team would reach it first."""

from dataclasses import dataclass, fields

import numpy as np

from tactifoot_vision.analysis.velocity import velocities
from tactifoot_vision.pipeline.postprocess import metres_per_unit
from tactifoot_vision.pipeline.result import PipelineResult
from tactifoot_vision.pitch.pitch import SoccerPitch

# Pitch control's integration over time
_DT = 0.04  # seconds a step
_START = -4  # intercept_sigmas before the cell's first arrival
_MAX_STEPS = 500  # 20 s
_UNCONTROLLED = 0.01  # share of every cell left free when it stops
_EPS = 1e-9

Team = tuple[np.ndarray, np.ndarray]  # players' positions (m) and velocities (m/s)


@dataclass(frozen=True, eq=False)
class SpaceControl:
    """Control of team 0 against team 1 over a grid of cells covering the pitch.

    ``control[row, column]`` is in ``[0, 1]``: ``1`` when team 0 controls the
    cell, ``0`` when team 1 does, ``0.5`` when neither does. Rows run along
    ``y`` and columns along ``x``, in the order of :attr:`y` and :attr:`x`,
    so the array reads like the radar (``x`` to the right, ``y`` down).
    """

    control: np.ndarray  # (rows, columns) float
    pitch: SoccerPitch

    def __post_init__(self) -> None:
        control = self.control
        if not isinstance(control, np.ndarray):
            raise ValueError(
                f"control must be a NumPy array, got {type(control).__name__}; "
                "pass np.asarray(values)"
            )
        if control.ndim != 2 or not control.size:
            raise ValueError(
                f"control must be a non-empty (rows, columns) array, got shape "
                f"{control.shape}"
            )
        if not ((control >= 0) & (control <= 1)).all():
            raise ValueError("control values must be in [0, 1]")

    @property
    def x(self) -> np.ndarray:
        """Pitch ``x`` of each column's cell centres."""
        return _centres(self.control.shape[1], self.pitch.length)

    @property
    def y(self) -> np.ndarray:
        """Pitch ``y`` of each row's cell centres."""
        return _centres(self.control.shape[0], self.pitch.width)


def space_control(
    result: PipelineResult,
    frames: int | slice | None = None,
    cell_size: float = 1.0,
    model: str = "nearest_player",
    **options: float,
) -> SpaceControl:
    """Space control of team 0 against team 1 by a model chosen by name.

    Every player with a team and a pitch position counts (players and
    goalkeepers alike); referees and people off the pitch plane do not.
    Distances are in metres and times in seconds whatever the pitch units.
    ``options`` are the chosen model's, each optional:

    ``model="nearest_player"``: the smoothed Voronoi partition. Each cell goes
    to the team whose nearest player is closer, softened across the frontier:
    the control is the sigmoid ``1 / (1 + exp(-(d1 - d0) / smoothing))`` of
    the gap between team 1's and team 0's nearest distances.

    * ``smoothing`` (1.2 m): width of the frontier between the teams;
      ``0`` gives the hard partition (``0.5`` on the frontier).

    ``model="pitch_control"``: Spearman's pitch control (Spearman et al.,
    2017, "Physics-Based Modeling of Pass Probabilities in Soccer"), without
    the ball. Each player keeps running with their velocity (from
    :func:`~tactifoot_vision.analysis.velocities`; none counts as standing)
    for the reaction time, then runs straight to the cell at top speed: that
    is their time to intercept. They reach the cell by time ``T`` with the
    probability ``1 / (1 + exp(-pi / sqrt(3) / intercept_sigma * (T - tti)))``
    and, once there, take control at ``control_rate`` per second while the
    cell is free. The control is team 0's share of the cell once it is
    almost surely taken. With no ball, every cell is contested as if the ball
    were there from the start, whatever the travel time of a pass; all
    players, goalkeepers too, share one reaction time, one top speed and one
    control rate, reach top speed at once and turn without slowing. The defaults are
    Spearman's and were not tuned on this package's data.

    * ``reaction_time`` (0.7 s): time a player keeps their velocity before
      changing course.
    * ``max_speed`` (5.0 m/s): speed of the straight run to the cell.
    * ``intercept_sigma`` (0.45 s): spread of the time to intercept, how
      uncertain an arrival is.
    * ``control_rate`` (4.3 per second): how fast a player who has arrived
      takes control (Spearman's lambda).
    * ``velocity_window`` (0.5 s): the ``window`` of
      :func:`~tactifoot_vision.analysis.velocities`, over the whole result.

    Args:
        frames: position in ``result.frames`` (as ``result[i]``, not the video
            frame index), a slice of them, or ``None`` for every frame. Over
            several frames the control is the mean of the frames' controls;
            frames without a positioned player in each team are left out.
        cell_size: side of a cell in metres (pitch units scaled to a
            105 x 68 m pitch). The cell counts along the length and the width
            are rounded so the cells tile the pitch exactly. Both models give
            the same grid.
        model: ``"nearest_player"`` or ``"pitch_control"``.

    Raises ``ValueError`` for an unknown model, an option the model does not
    take, and when no selected frame has a positioned player in each team.
    """
    if cell_size <= 0:
        raise ValueError(f"cell_size must be > 0, got {cell_size}")
    rule = _model(model, options)
    every = range(len(result.frames))
    selected = every[frames] if frames is not None else every
    selected = [selected] if isinstance(selected, int) else selected
    speeds = (
        velocities(result, rule.velocity_window)
        if isinstance(rule, _PitchControl)
        else None
    )
    pitch = result.pitch
    scale = metres_per_unit(pitch)
    columns, rows = (
        max(1, round(metres / cell_size))
        for metres in scale * [pitch.length, pitch.width]
    )
    cells = (
        np.stack(
            np.meshgrid(_centres(columns, pitch.length), _centres(rows, pitch.width)),
            axis=-1,
        )
        * scale
    )  # metres
    total, counted = np.zeros((rows, columns)), 0
    for position in selected:
        frame = result.frames[position]
        team_ids, xy = frame.team_ids, frame.pitch_xy * scale
        velocity = (
            np.nan_to_num(speeds[position]) if speeds is not None else np.zeros_like(xy)
        )
        positioned = np.isfinite(xy).all(axis=1)
        teams = [positioned & (team_ids == team) for team in (0, 1)]
        if not all(team.any() for team in teams):
            continue
        total += rule.control(cells, [(xy[team], velocity[team]) for team in teams])
        counted += 1
    if not counted:
        raise ValueError(
            "Space control needs positioned players of both teams 0 and 1 in a "
            f"frame; none in the {len(selected)} frame"
            f"{'' if len(selected) == 1 else 's'} selected. Select frames where "
            "both teams have pitch positions"
        )
    return SpaceControl(total / counted, pitch)


@dataclass(frozen=True)
class _NearestPlayer:
    """The nearest-player rule's options (see :func:`space_control`)."""

    smoothing: float = 1.2

    def __post_init__(self) -> None:
        if self.smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {self.smoothing}")

    def control(self, cells: np.ndarray, teams: list[Team]) -> np.ndarray:
        """Team 0's control of each cell (``cells``: (rows, columns, 2) metres)."""
        nearest = [_nearest(cells, xy) for xy, _ in teams]
        return _control(nearest[1] - nearest[0], self.smoothing)


@dataclass(frozen=True)
class _PitchControl:
    """The pitch-control model's options (see :func:`space_control`)."""

    reaction_time: float = 0.7
    max_speed: float = 5.0
    intercept_sigma: float = 0.45
    control_rate: float = 4.3
    velocity_window: float = 0.5

    def __post_init__(self) -> None:
        for name in ("reaction_time", "intercept_sigma"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("max_speed", "control_rate", "velocity_window"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")

    def control(self, cells: np.ndarray, teams: list[Team]) -> np.ndarray:
        """Team 0's control of each cell (``cells``: (rows, columns, 2) metres)."""
        shape = cells.shape[:2]
        arrivals = [
            self._intercept_times(cells.reshape(-1, 2), *team) for team in teams
        ]
        # Time runs from where the cell's first player could arrive: with no
        # ball, earlier times add nothing but the sigmoid's far tail.
        first = np.minimum(*(times.min(axis=0) for times in arrivals))
        arrivals = [(times - first).astype(np.float32) for times in arrivals]
        slope = float(np.pi / np.sqrt(3) / max(self.intercept_sigma, _EPS))
        taken, free = np.zeros((2, first.size)), np.ones(first.size)
        for step in range(_MAX_STEPS):
            if free.max() < _UNCONTROLLED:
                break
            time = _START * self.intercept_sigma + (step + 0.5) * _DT
            rates = np.stack([
                self.control_rate * _sigmoid(slope * (time - times)).sum(axis=0)
                for times in arrivals
            ])  # fmt: skip
            # The free share decays at the summed rate over the step, split
            # between the teams in proportion to their rates (the exact step
            # for constant rates, so no share overshoots however many arrive).
            rate = rates.sum(axis=0)
            gained = free * -np.expm1(-rate * _DT)
            taken += gained * np.divide(
                rates, rate, out=np.full_like(rates, 0.5), where=rate > 0
            )
            free -= gained
        both = taken[0] + taken[1]
        control = np.divide(taken[0], both, out=np.full_like(both, 0.5), where=both > 0)
        return control.reshape(shape)

    def _intercept_times(
        self, cells: np.ndarray, xy: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        """Each player's time to intercept at each of ``cells`` (N, 2), (players, N) s."""
        turn = xy + velocity * self.reaction_time
        distance = np.hypot(
            cells[:, 0] - turn[:, 0, None], cells[:, 1] - turn[:, 1, None]
        )
        return self.reaction_time + distance / self.max_speed


_MODELS: dict[str, type[_NearestPlayer] | type[_PitchControl]] = {
    "nearest_player": _NearestPlayer,
    "pitch_control": _PitchControl,
}


def _model(name: str, options: dict[str, float]) -> _NearestPlayer | _PitchControl:
    """The named model with ``options``."""
    if name not in _MODELS:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(_MODELS)}")
    model = _MODELS[name]
    valid = [field.name for field in fields(model)]
    unknown = sorted(set(options) - set(valid))
    if unknown:
        raise ValueError(
            f"Model {name!r} takes {', '.join(valid)}; got {', '.join(unknown)}"
        )
    return model(**options)


def _centres(n: int, size: float) -> np.ndarray:
    """Centres of ``n`` equal cells tiling ``[0, size]``."""
    return (np.arange(n) + 0.5) * size / n


def _nearest(cells: np.ndarray, players: np.ndarray) -> np.ndarray:
    """Distance from each cell to the nearest of ``players``, one player at a time."""
    nearest = np.full(cells.shape[:2], np.inf)
    for x, y in players:
        np.minimum(nearest, np.hypot(cells[..., 0] - x, cells[..., 1] - y), out=nearest)
    return nearest


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """The logistic sigmoid as tanh, which does not overflow."""
    return 0.5 * (1 + np.tanh(x / 2))


def _control(lead: np.ndarray, smoothing: float) -> np.ndarray:
    """Team 0's control from how much nearer it is (metres), in ``[0, 1]``."""
    if smoothing == 0:
        return 0.5 * (1 + np.sign(lead))
    return _sigmoid(lead / smoothing)
