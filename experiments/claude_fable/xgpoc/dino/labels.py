"""Header labels from SoccerNet-Caption commentary and the GPT sample (ticket 10, spec §Attributes Xl, council C3.4).

Labels are probe targets and subgroup keys only: they carry a `source` ("caption" or "gpt") and `assert_not_design`
refuses them as a model-design column. A comment mentions the outcome ("Goal!"), so the outcome is never used to match:
the shot's own SoccerNet-v2 annotation (half, position, team) and the comment's text are the only inputs of the match.
`goal` is read by the caller for the coverage tables and the sample strata, never by `match_game`.

Caption files
    `Labels-caption.json` of the training matches, from the public ungated HF dataset `SoccerNet/SN-Labels`, to
    /mnt/storage/tactifoot_research/xg_dino/info/captions/<game>/Labels-caption.json. The comments were scraped from
    flashscore; their terms are unchecked (the dataset card has no licence). 25 of 328 training matches have no file.
Match (`match_game`), per game
    candidate pair   same half; comment time - shot time in [-5 s, +60 s]; the comment is shot-like (`is_shot_like`, outcome-
                     neutral words only); if the comment names a team in brackets and it is the other team of the match, no pair
    ambiguity        a comment that is a candidate of two or more shots is rejected for all of them (those shots: unknown,
                     reason "ambiguous"); shots with several comments keep the one nearest to the median lag (`LAG_MS`)
                     and fall to unknown when their comments disagree between header and foot
    label            header  header words, after removing defender phrases ("headed clear", "headed away", ...)
                     foot    foot words and no header words; no header words is not enough
                     unknown shot-like comment with neither, no comment, ambiguity, or both kinds of words
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.stack import inputs as si

CAPTION_REPO = "SoccerNet/SN-Labels"
CAPTION_FILE = "Labels-caption.json"
INFO = Path("/mnt/storage/tactifoot_research/xg_dino/info")
CAPTION_DIR = INFO / "captions"
WINDOW_MS = (-5000, 60000)
LAG_MS = 12000                                  # the median comment lag of the 25-game probe (recon-external.md)
LABELS = ("header", "foot", "unknown")
SOURCES = ("caption", "gpt")
LABEL_COLUMNS = ("header_caption", "header_gpt")

SHOT_WORDS = re.compile(r"\b(shots?|shoots?|shooting|strikes?|striking|effort|volley|volleys|volleyed|header|headers|heads|headed|"
                        r"attempts?|fires|fired|drills?|drilled|curls?|curled|blasts?|lobs?|chips?|drives?|nods?|flicks?|"
                        r"touch(?:es)? the ball past)\b", re.I)
HEADER_WORDS = re.compile(r"\b(header|headers|headed|heads|nods?|nodded|with (?:his|her) head|glancing header)\b", re.I)
FOOT_WORDS = re.compile(r"\b(left[- ]foot(?:ed)?|right[- ]foot(?:ed)?|(?:left|right)[- ]footer|foot|volley|volleys|volleyed|"
                        r"half[- ]volley|toe[- ]poke|toe|instep|boot|with (?:his|her) (?:left|right))\b", re.I)
DEFENDER_PHRASES = re.compile(r"\b(head(?:s|ed)? (?:it )?(?:clear|away|out|back|off the line)|(?:header|headed) (?:clear|away)|"
                              r"clears? (?:the ball )?with (?:a|his|her) head|heads? (?:the ball )?(?:clear|away)|"
                              r"cleared? with (?:a|his|her) header)\b", re.I)
TEAM_IN_BRACKETS = re.compile(r"\(([^()]+)\)")


def caption_path(game: str, root: Path = CAPTION_DIR) -> Path:
    return Path(root) / game / CAPTION_FILE


def check_reserved(game: str) -> None:
    """Refuse a game path that ntfs3 / Windows cannot hold (reserved device names, trailing dot or space, illegal chars)."""
    for part in game.split("/"):
        if re.search(r'[<>:"|?*\\]', part) or part != part.rstrip(". ") or part.split(".")[0].upper() in {
                "CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
            raise ValueError(f"game path part {part!r} is not safe on ntfs3")


def download_captions(games, root: Path = CAPTION_DIR) -> dict[str, str]:
    """Download `Labels-caption.json` of every game in `games` that the dataset has; {game: "ok" | "absent"}."""
    from huggingface_hub import HfApi, hf_hub_download
    on_hub = {f.rsplit("/", 1)[0] for f in HfApi().list_repo_files(CAPTION_REPO, repo_type="dataset") if f.endswith(CAPTION_FILE)}
    status = {}
    for game in sorted(games):
        check_reserved(game)
        if game not in on_hub:
            status[game] = "absent"
            continue
        if not caption_path(game, root).exists():
            hf_hub_download(CAPTION_REPO, f"{game}/{CAPTION_FILE}", repo_type="dataset", local_dir=str(root))
        status[game] = "ok"
    return status


# ------------------------------------------------------------------ text rules
def is_shot_like(text: str) -> bool:
    return bool(SHOT_WORDS.search(DEFENDER_PHRASES.sub(" ", text)))


def comment_label(text: str) -> str:
    """header / foot / unknown for a comment, after removing the defender phrases (module docstring)."""
    clean = DEFENDER_PHRASES.sub(" ", text)
    head, foot = bool(HEADER_WORDS.search(clean)), bool(FOOT_WORDS.search(clean))
    return "header" if head and not foot else "foot" if foot and not head else "unknown"


@dataclass(frozen=True)
class Comment:
    half: int
    position_ms: int
    text: str
    team: str | None            # "home" / "away" when the comment names one of the two teams in brackets


def read_comments(path: Path) -> list[Comment]:
    """The shot-like comments of a caption file (`Labels-caption.json`); the team from the first bracket naming a team."""
    raw = json.loads(Path(path).read_text())
    names = {"home": {n.lower() for n in raw.get("home", {}).get("names", [])} | {str(raw.get("gameHomeTeam", "")).lower()},
             "away": {n.lower() for n in raw.get("away", {}).get("names", [])} | {str(raw.get("gameAwayTeam", "")).lower()}}
    out = []
    for a in raw["annotations"]:
        text = a.get("description", "")
        if not is_shot_like(text):
            continue
        team = next((side for b in TEAM_IN_BRACKETS.findall(text) for side in names if b.strip().lower() in names[side]), None)
        out.append(Comment(int(a["gameTime"].split(" - ")[0]), int(a["position"]), text, team))
    return out


def shot_teams(annotations: list[dict], shots: pd.DataFrame) -> dict[str, str | None]:
    """{shot_id: home / away / None} from the shot's own SoccerNet-v2 annotation (same half and position)."""
    parsed = si.parse_annotations(annotations)
    out = {}
    for r in shots.itertuples():
        teams = {a.team for a in parsed if a.kind == si.SHOT and a.half == r.half and a.position_ms == r.position_ms}
        out[r.Index] = teams.pop() if len(teams) == 1 else None
    return out


def match_game(shots: pd.DataFrame, teams: dict[str, str | None], comments: list[Comment]) -> pd.DataFrame:
    """One row per shot of one game (index shot_id; half, position_ms): `label`, `reason`, `lag_ms`, `n_candidates`.
    No outcome column goes in; nothing here can read one."""
    lo, hi = WINDOW_MS
    cand: dict[str, list[int]] = {}
    for r in shots.itertuples():
        cand[r.Index] = [j for j, c in enumerate(comments)
                         if c.half == r.half and lo <= c.position_ms - r.position_ms <= hi
                         and (c.team is None or teams[r.Index] is None or c.team == teams[r.Index])]
    owners: dict[int, list[str]] = {}
    for s, js in cand.items():
        for j in js:
            owners.setdefault(j, []).append(s)
    rows = []
    for r in shots.itertuples():
        js = cand[r.Index]
        free = [j for j in js if len(owners[j]) == 1]
        if not js:
            label, reason, lag = "unknown", "no_comment", np.nan
        elif not free:
            label, reason, lag = "unknown", "ambiguous", np.nan
        else:
            kinds = {comment_label(comments[j].text) for j in free} - {"unknown"}
            best = min(free, key=lambda j: abs(comments[j].position_ms - r.position_ms - LAG_MS))
            lag = comments[best].position_ms - r.position_ms
            if len(kinds) == 2:
                label, reason = "unknown", "conflict"
            elif kinds:
                label, reason = kinds.pop(), "matched"
            else:
                label, reason = "unknown", "no_body_part_words"
        rows.append({"shot_id": r.Index, "label": label, "reason": reason, "lag_ms": lag, "n_candidates": len(js)})
    return pd.DataFrame(rows).set_index("shot_id")


def caption_labels(shots: pd.DataFrame, labels_root: Path, caption_root: Path = CAPTION_DIR) -> pd.DataFrame:
    """`header_caption` table for `shots` (index shot_id; game, half, position_ms): columns label, reason, lag_ms,
    n_candidates, source="caption". A game without a caption file gets reason "no_caption_file"."""
    parts = []
    for game, g in shots.groupby("game", sort=True):
        path = caption_path(game, caption_root)
        if not path.exists():
            parts.append(pd.DataFrame({"label": "unknown", "reason": "no_caption_file", "lag_ms": np.nan, "n_candidates": 0},
                                      index=g.index))
            continue
        teams = shot_teams(si.read_annotations(game, labels_root), g)
        parts.append(match_game(g, teams, read_comments(path)))
    out = pd.concat(parts).loc[shots.index]
    out["source"] = "caption"
    return out


def assert_not_design(columns) -> None:
    """Raise if any of `columns` is a caption or GPT label (they are probe targets and subgroup keys, not features)."""
    bad = [c for c in columns if str(c) in LABEL_COLUMNS or re.search(r"(header|body_part)_(caption|gpt)|_(caption|gpt)$", str(c))]
    if bad:
        raise ValueError(f"caption / GPT labels are not allowed in a model design: {bad}")


# ------------------------------------------------------------------ the GPT sample
SAMPLE_SIZE = 1000
SAMPLE_SEED = 20261028
DISTANCE_BANDS_M = (12.0, 20.0)                 # < 12, 12-20, >= 20; a missing distance is its own band "na"
OFFSETS = {0: "t0", -400: "m400"}
SCOREBOARD_FRACTION = 0.15                      # top band blanked: every broadcast overlay of the sample sits inside it
GPT_DIR = INFO / "gpt_frames"
FRAMES_DIR = GPT_DIR / "frames"
KEY_DIR = GPT_DIR / "private_key"
ANSWERS = ("head", "foot", "other", "unclear")


def distance_band(dist: pd.Series) -> pd.Series:
    lo, hi = DISTANCE_BANDS_M
    return pd.Series(np.where(dist.isna(), "na", np.where(dist < lo, "lt12", np.where(dist < hi, "12to20", "ge20"))), index=dist.index)


def sample_allocation(sizes: dict, total: int) -> dict:
    """Equal share per stratum, capped by its size, the remainder spread over the unfilled strata (water filling); the
    ties of the last round go to the larger stratum, then to the key order. Sums to min(total, sum(sizes))."""
    take = dict.fromkeys(sizes, 0)
    left = min(total, sum(sizes.values()))
    while left > 0:
        open_ = [k for k in sizes if take[k] < sizes[k]]
        share = max(left // len(open_), 1)
        for k in sorted(open_, key=lambda k: (-(sizes[k] - take[k]), str(k))):
            add = min(share, sizes[k] - take[k], left)
            take[k] += add
            left -= add
    return take


def select_sample(table: pd.DataFrame, total: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """`table` (index shot_id; goal, caption_label, dist) -> the sample rows with `stratum` and `weight` (stratum size /
    stratum sample). Strata: outcome x caption label x distance band, seeded draw within each."""
    t = pd.DataFrame({"outcome": table.goal.astype(int), "caption_label": table.caption_label,
                      "band": distance_band(table.dist)}, index=table.index)
    t["stratum"] = t.outcome.astype(str) + "|" + t.caption_label + "|" + t.band
    sizes = t.stratum.value_counts().sort_index().to_dict()
    take = sample_allocation(sizes, total)
    rng = np.random.default_rng(seed)
    picked = []
    for stratum, n in sorted(take.items()):
        ids = np.sort(t.index[t.stratum == stratum].to_numpy())
        picked += list(rng.choice(ids, size=n, replace=False))
    out = t.loc[sorted(picked)].copy()
    out["weight"] = out.stratum.map(sizes) / out.stratum.map(out.stratum.value_counts())
    return out


def anon_ids(shot_ids, seed: int = SAMPLE_SEED) -> pd.Series:
    """A seeded random id per shot (`s` + 5 digits, no relation to the shot id, the order or the stratum)."""
    ids = sorted(shot_ids)
    codes = np.random.default_rng(seed + 1).permutation(100000)[:len(ids)]
    return pd.Series([f"s{c:05d}" for c in codes], index=ids)


def blank_scoreboard(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[:int(np.ceil(out.shape[0] * SCOREBOARD_FRACTION))] = 0
    return out


def write_frames(sample_ids, frames: pd.DataFrame, ids: pd.Series, out_dir: Path = FRAMES_DIR) -> int:
    """Write `<anon_id>_{t0,m400}.jpg` for every shot (scoreboard band blanked, no metadata); `frames` has shot_id,
    offset_ms, frame_path. Returns the number of files written; raises on an unreadable frame."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = frames[frames.offset_ms.isin(OFFSETS)].set_index(["shot_id", "offset_ms"]).frame_path
    n = 0
    for shot_id in sample_ids:
        for offset, name in OFFSETS.items():
            image = cv2.imread(str(paths.loc[(shot_id, offset)]))
            if image is None:
                raise ValueError(f"{shot_id} at {offset} ms: frame not readable")
            if not cv2.imwrite(str(out_dir / f"{ids[shot_id]}_{name}.jpg"), blank_scoreboard(image), [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise ValueError(f"could not write {ids[shot_id]}_{name}.jpg")
            n += 1
    return n


INSTRUCTIONS = """\
# Labelling the body part of a shot

Each item is a football shot taken from a broadcast. For every id `sNNNNN` there are two still frames:

- `sNNNNN_m400.jpg`: 0.4 seconds before the moment of the shot,
- `sNNNNN_t0.jpg`: the moment of the shot (the ball is struck at about this frame).

The top strip of each image is blacked out on purpose. Use only what you see in the two frames. Nothing else is known
about the shot, and you must not try to find out more.

## Question

With which body part does the shooter strike the ball at the moment of the shot? The shooter is the player who is
about to strike, or is striking, the ball towards the goal.

Answer with exactly one of:

- `head`: the shooter heads the ball (the head is the striking part, or clearly about to be).
- `foot`: the shooter strikes the ball with a foot, including a volley, a half-volley, a toe poke or a slide.
- `other`: some other body part (chest, shoulder, knee, thigh, hand), or a deflection that is not a deliberate strike.
- `unclear`: you cannot tell from the two frames (shooter not visible, too small, ball not in play, a cut or a replay,
  or the two frames show different scenes). Use `unclear` rather than guess.

## Output

One line per id, in this exact format, nothing else:

    sNNNNN,head
    sNNNNN,foot
    sNNNNN,other
    sNNNNN,unclear

Give an answer for every id in the folder. Do not guess the result of the shot and do not answer about it.
"""
