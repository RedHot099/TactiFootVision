"""Attack direction: which goal each team attacks, from where its players stand."""

import numpy as np

from tactifoot_vision.pipeline.postprocess import metres_per_unit
from tactifoot_vision.pipeline.result import NO_TEAM, PipelineResult


def attack_direction(
    result: PipelineResult, min_separation: float = 1.0
) -> dict[int, int]:
    """The direction each team attacks in, by team id: ``1`` or ``-1``.

    ``1`` is towards the goal at ``x = result.pitch.length``, ``-1`` towards
    the goal at ``x = 0``. A run covers one period (its ``period`` run input),
    so call this on each period's result: the directions flip at half-time.

    Each team's mean pitch ``x`` over every frame (every person with a team
    and a pitch position: players and goalkeepers) says where it stands; the
    team nearer ``x = 0`` defends that goal and attacks the other one. The
    teams are compared with each other, not with the halfway line, so a
    camera that stays in one half does not turn both teams the same way.

    ``min_separation`` is the smallest gap, in metres along the length (pitch
    units scaled to a 105 m pitch), between the two means that decides the
    direction; a smaller gap is too close to call.

    Raises ``ValueError`` unless exactly two teams have a pitch position, or
    when their means are less than ``min_separation`` apart.
    """
    if min_separation < 0:
        raise ValueError(f"min_separation must be >= 0, got {min_separation}")
    xs: dict[int, list[np.ndarray]] = {}
    for frame in result:
        team_ids, pitch_x = frame.team_ids, frame.pitch_xy[:, 0]
        for team in np.unique(team_ids[team_ids != NO_TEAM]):
            x = pitch_x[(team_ids == team) & np.isfinite(pitch_x)]
            if len(x):
                xs.setdefault(int(team), []).append(x)
    if len(xs) != 2:
        found = ", ".join(map(str, sorted(xs))) or "none"
        teams = "team" if len(xs) == 1 else "teams"
        raise ValueError(
            f"The attack direction needs two teams with pitch positions; "
            f"the result has {teams} {found}"
        )
    means = {team: float(np.concatenate(x).mean()) for team, x in xs.items()}
    defends_x0, defends_length = sorted(means, key=means.__getitem__)
    gap = (means[defends_length] - means[defends_x0]) * metres_per_unit(result.pitch)[0]
    if gap == 0 or gap < min_separation:
        raise ValueError(
            f"Teams {defends_x0} and {defends_length} stand {gap:.2f} m apart in "
            f"mean pitch x, under min_separation={min_separation} m; the attack "
            "direction is undecided"
        )
    return {defends_x0: 1, defends_length: -1}
