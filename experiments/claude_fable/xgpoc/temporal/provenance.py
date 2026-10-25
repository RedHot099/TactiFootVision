"""Leak assertions of xg-temporal (ticket 01; spec, "Kontrola przecieku" 1, 3 and 4). Each check raises
`ProvenanceError`; `control` runs them all on a population and its arms and records which passed.

    window    o_max > -600 ms (`LEAK_LIMIT`) is a leakage-control row: allowed only with the explicit control flag,
              never a candidate
    columns   every design column but the reference logit (`REFERENCE`) is named `<feature>@<offset>`, the offset being
              the latest frame the column uses; no offset > o_max, no anchor frame (offset 0) outside a control row,
              no banned fragment (`BANNED`, case-insensitive) in any name
    frames    every frame row that fed a feature is at an offset <= o_max (the anchor only in a control row), decoded
              (frame_status "ok"), without a cut between it and the anchor, and has an embedding
    roles     every match of the population is a `training` match of the split plan, none of the 172 held-out ones
"""

import pandas as pd

BANNED = ("goal", "label", "outcome", "statsbomb", "sb_")
LEAK_LIMIT = -600
ANCHOR = 0
REFERENCE = "z_g_combo"
ROLE = "training"
HELD_OUT_ROLES = ("evaluation", "gate")
OK = "ok"                                       # frame_status of a decoded frame


class ProvenanceError(AssertionError):
    """A leak assertion of xg-temporal failed."""


def uncut(frames: pd.DataFrame) -> pd.Series:
    """No cut between the frame and the anchor; a missing cut flag (no frame) counts as a cut."""
    return ~frames.cut_between_anchor.astype("boolean").fillna(True).astype(bool)


def check_window(o_max: int, control: bool) -> None:
    if o_max > LEAK_LIMIT and not control:
        raise ProvenanceError(f"o_max {o_max} > {LEAK_LIMIT} ms is a leakage-control row and needs the control flag")


def check_names(columns) -> None:
    for c in columns:
        if found := [f for f in BANNED if f in str(c).lower()]:
            raise ProvenanceError(f"column {c!r} contains the banned fragment(s) {found}")


def column_offset(name: str) -> int:
    """The offset of a `<feature>@<offset>` column name."""
    feature, sep, offset = name.rpartition("@")
    try:
        if not sep or not feature:
            raise ValueError
        return int(offset)
    except ValueError:
        raise ProvenanceError(f"column {name!r} is not named <feature>@<offset>") from None


def check_offset(offset: int, o_max: int, control: bool, what: str) -> None:
    if offset > o_max:
        raise ProvenanceError(f"{what} uses offset {offset} > o_max {o_max}")
    if offset == ANCHOR and not control:
        raise ProvenanceError(f"{what} uses the anchor frame")


def check_columns(columns, o_max: int, control: bool = False) -> None:
    """The design column names of an arm (module docstring)."""
    check_window(o_max, control)
    check_names(columns)
    for c in columns:
        if c != REFERENCE:
            check_offset(column_offset(c), o_max, control, f"column {c!r}")


def check_frames(frames: pd.DataFrame, o_max: int, control: bool = False) -> None:
    """The frame rows with `used` True (offset_ms, frame_status, cut_between_anchor, embedding_ref): module docstring."""
    check_window(o_max, control)
    used = frames[frames.used.astype(bool)]
    for offset in sorted(used.offset_ms.unique()):
        check_offset(int(offset), o_max, control, "a used frame")
    if (bad := used.frame_status != OK).any():
        raise ProvenanceError(f"{int(bad.sum())} used frames are not decoded frames")
    if (cut := ~uncut(used)).any():
        raise ProvenanceError(f"{int(cut.sum())} used frames have a cut between them and the anchor (or no cut flag)")
    if (used.embedding_ref < 0).any():
        raise ProvenanceError("a used frame has no embedding")


def check_roles(games, plan: pd.DataFrame) -> None:
    """Every game is a training match of `plan` and none is held out."""
    role = plan.set_index("game").role
    games = pd.Series(sorted(set(games)), dtype=object)
    if not (unknown := games[~games.isin(role.index)]).empty:
        raise ProvenanceError(f"{len(unknown)} population matches are not in the plan, e.g. {unknown.iloc[0]}")
    roles = games.map(role)
    if not (held := games[roles.isin(HELD_OUT_ROLES)]).empty:
        raise ProvenanceError(f"{len(held)} population matches are held out, e.g. {held.iloc[0]}")
    if not (other := games[roles != ROLE]).empty:
        raise ProvenanceError(f"{len(other)} population matches are outside the {ROLE} role, e.g. {other.iloc[0]}")


def control(population, columns: dict[str, list[str]], plan: pd.DataFrame) -> dict:
    """Every check (module docstring) on a `inputs.Population` and its arms' design columns:
    {"passed", "control_row", "checks": [{"check", "passed", "detail"}]}."""
    window = population.window
    checks = []

    def run(name: str, fn, *args) -> None:
        try:
            fn(*args)
            checks.append({"check": name, "passed": True, "detail": ""})
        except ProvenanceError as e:
            checks.append({"check": name, "passed": False, "detail": str(e)})

    run("window", check_window, window.o_max, window.control)
    for arm, names in columns.items():
        run(f"columns:{arm}", check_columns, names, window.o_max, window.control)
    run("frames", check_frames, population.frames, window.o_max, window.control)
    run("roles", check_roles, population.games, plan)
    return {"passed": all(c["passed"] for c in checks), "control_row": window.control, "checks": checks}
