from collections.abc import Iterable

FORBIDDEN_INPUT_COLUMNS = frozenset(
    {
        "location",
        "shot_location",
        "shot_freeze_frame",
        "freeze_frame",
        "shot_statsbomb_xg",
        "statsbomb_xg",
        "under_pressure",
        "shot_body_part",
        "shot_type",
        "shot_technique",
        "shot_first_time",
        "shot_one_on_one",
        "shot_outcome",
        "outcome",
        "goal",
        "is_goal",
    }
)


class ForbiddenVideoXgInputError(ValueError):
    pass


def assert_video_only_columns(columns: Iterable[str]) -> None:
    normalized = {column.strip().lower() for column in columns}
    forbidden = sorted(normalized & FORBIDDEN_INPUT_COLUMNS)
    if forbidden:
        joined = ", ".join(forbidden)
        raise ForbiddenVideoXgInputError(
            f"Video-only xG features cannot include StatsBomb/label columns: {joined}"
        )
