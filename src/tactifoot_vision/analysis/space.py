"""Space control: for each point of the pitch, which team would reach it first."""

from dataclasses import dataclass

import numpy as np

from tactifoot_vision.pipeline.postprocess import metres_per_unit
from tactifoot_vision.pipeline.result import PipelineResult
from tactifoot_vision.pitch.pitch import SoccerPitch


@dataclass(frozen=True)
class SpaceControl:
    """Control of team 0 against team 1 over a grid of cells covering the pitch.

    ``control[row, column]`` is in ``[0, 1]``: ``1`` when team 0 controls the
    cell, ``0`` when team 1 does, ``0.5`` when neither does. Rows run along
    ``y`` and columns along ``x``, in the order of :attr:`y` and :attr:`x`,
    so the array reads like the radar (``x`` to the right, ``y`` down).
    """

    control: np.ndarray  # (rows, columns) float
    pitch: SoccerPitch

    @property
    def x(self) -> np.ndarray:
        """Pitch ``x`` of each column's cell centres."""
        n = self.control.shape[1]
        return (np.arange(n) + 0.5) * self.pitch.length / n

    @property
    def y(self) -> np.ndarray:
        """Pitch ``y`` of each row's cell centres."""
        n = self.control.shape[0]
        return (np.arange(n) + 0.5) * self.pitch.width / n


def space_control(
    result: PipelineResult,
    frames: int | slice | None = None,
    cell_size: float = 1.0,
    smoothing: float = 1.2,
) -> SpaceControl:
    """Space control of team 0 against team 1 by the nearest-player rule.

    Each cell goes to the team whose nearest player (players and goalkeepers:
    everyone with a team and a pitch position) is closer, softened across the
    frontier: the control is the sigmoid ``1 / (1 + exp(-(d1 - d0) / smoothing))``
    of the gap between team 1's and team 0's nearest distances, in metres.
    ``smoothing=0`` gives the hard Voronoi partition (``0.5`` on the frontier).

    Args:
        frames: position in ``result.frames`` (as ``result[i]``), a slice of
            them, or ``None`` for every frame. Over several frames the control
            is the mean of the frames' controls; frames without a positioned
            player in each team are left out.
        cell_size: side of a cell in metres (pitch units scaled to a
            105 x 68 m pitch). The cell counts along the length and the width
            are rounded so the cells tile the pitch exactly.
        smoothing: width of the frontier between the teams, in metres.

    Raises ``ValueError`` when no selected frame has a positioned player in
    each team.
    """
    if cell_size <= 0:
        raise ValueError(f"cell_size must be > 0, got {cell_size}")
    if smoothing < 0:
        raise ValueError(f"smoothing must be >= 0, got {smoothing}")
    selected = result.frames[frames] if frames is not None else result.frames
    selected = selected if isinstance(selected, list) else [selected]
    pitch = result.pitch
    scale = metres_per_unit(pitch)
    columns, rows = (
        max(1, round(metres / cell_size))
        for metres in scale * [pitch.length, pitch.width]
    )
    grid = SpaceControl(np.empty((rows, columns)), pitch)
    cells = np.stack(np.meshgrid(grid.x, grid.y), axis=-1) * scale  # metres
    controls = []
    for frame in selected:
        team_ids, xy = frame.team_ids, frame.pitch_xy * scale
        positioned = np.isfinite(xy).all(axis=1)
        nearest = []
        for team in (0, 1):
            players = xy[positioned & (team_ids == team)]
            if not len(players):
                break
            gaps = cells[:, :, None, :] - players
            nearest.append(np.hypot(gaps[..., 0], gaps[..., 1]).min(axis=-1))
        else:
            controls.append(_control(nearest[1] - nearest[0], smoothing))
    if not controls:
        raise ValueError(
            "Space control needs positioned players of both teams 0 and 1 in a "
            f"frame; none in the {len(selected)} frame"
            f"{'' if len(selected) == 1 else 's'} selected"
        )
    return SpaceControl(np.mean(controls, axis=0), pitch)


def _control(lead: np.ndarray, smoothing: float) -> np.ndarray:
    """Team 0's control from how much nearer it is (metres), in ``[0, 1]``."""
    if smoothing == 0:
        return 0.5 * (1 + np.sign(lead))
    # The logistic sigmoid as tanh, which does not overflow.
    return 0.5 * (1 + np.tanh(lead / (2 * smoothing)))
