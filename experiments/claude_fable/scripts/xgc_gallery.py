"""Local gallery of the largest frame - StatsBomb disagreements, one shot per category (owner request, 2026-10-14).

    ./run.sh --no-sync python scripts/xgc_gallery.py

For the owner's own look at the frames; agents do not look at them (council C4, Q4 a). From the top-5% disagreement
list of ticket 02 (results/xgc_compare/analysis/disagreement_shots.csv) it takes, for each category of `CATEGORIES` in
order, the shot with the largest |F - V| in the category's direction that no earlier category took. Each card shows:

- the index frames at -800, -400, 0, +400 ms and the 0 ms frame with the detections of the xg-detector cache
  (cache/xgd_full3/E3: players by single-frame team, goalkeepers, ball);
- the attacked penalty area in aligned metres (`xgpoc.detector.conventions`): the StatsBomb freeze frame (filled) beside
  the detected positions (hollow);
- every method's probability, the outcome, and the StatsBomb geometry beside the detector's, both in metres.

Writes results/xgc_compare/gallery (untracked: it holds SoccerNet frames and StatsBomb data, which are never committed or
published). Data: StatsBomb.
"""

import argparse
import html
import json
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from xgpoc.detector import audit
from xgpoc.detector.conventions import POSTS, align, statsbomb_to_metres
from xgpoc.detector.features import geometry
from xgpoc.paths import INDEX

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DISAGREEMENT = ROOT / "results/xgc_compare/analysis/disagreement_shots.csv"
TABLE = ROOT / "results/xgc_compare/table/shots.parquet"
E3 = ROOT / "cache/xgd_full3/E3"
OUT = ROOT / "results/xgc_compare/gallery"
OFFSETS = ("-800", "-400", "+0", "+400")
ABOVE, BELOW = "F_above_V", "F_below_V"
METHODS = [("p_V", "StatsBomb (V)"), ("p_F", "obraz (F)"), ("p_F-400", "obraz −400 ms"), ("p_A1", "obraz + detektor (A1)"),
           ("p_D", "odległość + kąt (D)"), ("p_P", "pozycje (P)"), ("p_E", "zdarzenia (E)")]

# slug, label, rule on the disagreement rows, direction (None: either), what the report found
CATEGORIES = [
    ("gk_far", "Bramkarz wysunięty ≥ 4 jd", lambda d: d.cat_gk_depth == "04_plus", BELOW,
     "obraz zaniża (O/E 1,26 wobec 0,99); odzyskuje ~39% wiedzy StatsBomb"),
    ("pressure", "Strzał pod presją", lambda d: d.cat_under_pressure == "yes", ABOVE,
     "obraz zawyża o ~1,5 pkt xG; ΔLL +0,037 na korzyść StatsBomb"),
    ("outside_box", "Spoza pola karnego", lambda d: d.cat_outside_box == "yes", ABOVE,
     "obraz zawyża: 68,1 przewidzianych goli wobec 57 strzelonych"),
    ("sharp_angle", "Ostry kąt < 15°", lambda d: d.cat_angle == "00_15", ABOVE,
     "obraz zawyża ponad dwukrotnie: 52,0 przewidzianych goli wobec 24 (skan, replikacja)"),
    ("header_below", "Główka (obraz niżej)", lambda d: d.cat_body == "Head", BELOW,
     "obraz odzyskuje ~48% wiedzy StatsBomb o główkach; słabiej rozróżnia (AUC 0,62 wobec 0,69)"),
    ("header_above", "Główka (obraz wyżej)", lambda d: d.cat_body == "Head", ABOVE,
     "jak wyżej, przypadek w drugą stronę"),
    ("cross", "Po dośrodkowaniu", lambda d: d.cat_key_pass_cross == "yes", BELOW,
     "słaby efekt (+0,135, tylko BH); w ogonie 5% głównie obraz niżej"),
    ("rebound", "Dobitka (≤ 5 s)", lambda d: d.cat_rebound_5s == "yes", BELOW,
     "Twoje zdanie (b): nie wykazano (moc 0,35)"),
    ("free_kick", "Rzut wolny bezpośredni", lambda d: d.cat_shot_type == "Free Kick", ABOVE,
     "22 ze 136 strzałów ogona „obraz wyżej”"),
    ("close", "Z bliska < 6 jd", lambda d: d.cat_distance == "00_06", BELOW,
     "30 ze 136 strzałów ogona „obraz niżej”"),
    ("wide_angle", "Duży kąt ≥ 45°", lambda d: d.cat_angle == "45_plus", BELOW,
     "55 ze 136 strzałów ogona „obraz niżej”"),
    ("long_on_target", "Z dystansu, w światło bramki", lambda d: (d.cat_outside_box == "yes") & (d.cat_on_target_sn == "yes"),
     ABOVE, "Twoje zdanie (a): nie potwierdzone (+0,047)"),
    ("one_on_one", "Sam na sam", lambda d: d.cat_one_on_one == "yes", None, "pozycja 12 rodziny: nierozstrzygnięta"),
    ("open_goal", "Pusta bramka", lambda d: d.cat_open_goal == "yes", None, "pozycja 6b: tylko opisowa"),
    ("tight_view", "Ciasny kadr (mało graczy)", lambda d: d.cat_players_visible == "low", None,
     "pozycja 10 (log-loss): praktycznie ślepa"),
    ("no_ball", "Detektor nie widzi piłki", lambda d: d.cat_ball_detected == "no", None, "jakość wejścia, opisowo"),
]
TAGS = [("cat_body", None), ("cat_technique", None), ("cat_shot_type", None), ("cat_play_pattern", None),
        ("cat_under_pressure", "presja"), ("cat_first_time", "z pierwszej"), ("cat_one_on_one", "sam na sam"),
        ("cat_open_goal", "pusta bramka"), ("cat_key_pass_cross", "dośrodkowanie"), ("cat_rebound_5s", "dobitka"),
        ("cat_outside_box", "spoza pola"), ("cat_gk_depth", "bramkarz [jd]"), ("cat_distance", "odległość [jd]"),
        ("cat_angle", "kąt [°]"), ("cat_players_visible", "graczy w kadrze"), ("cat_on_target_sn", "w światło (SN)")]


def pick(dis: pd.DataFrame) -> list[tuple[tuple, pd.Series]]:
    """One row per category: the largest |gap| in the category's direction among the shots not taken yet."""
    taken, out = set(), []
    for cat in CATEGORIES:
        _, _, rule, direction, _ = cat
        rows = dis[rule(dis) & ~dis.key.isin(taken)]
        if direction is not None:
            rows = rows[rows.direction == direction]
        if rows.empty:
            print(f"no shot left for {cat[0]}")
            continue
        row = rows.loc[rows.gap_F_minus_V.abs().idxmax()]
        taken.add(row.key)
        out.append((cat, row))
    return out


def statsbomb_people(freeze: str | None) -> list[dict]:
    """Freeze-frame people in aligned metres with their role."""
    people = json.loads(freeze) if isinstance(freeze, str) else []
    out = []
    for p in people:
        gk = (p.get("position") or {}).get("name") == audit.GOALKEEPER_POSITION
        role = "teammate" if p["teammate"] else ("gk" if gk else "opponent")
        out.append({"xy": statsbomb_to_metres(p["location"][:2]), "role": role})
    return out


def statsbomb_geometry(shot_xy: np.ndarray, people: list[dict]) -> dict:
    if not people:
        return geometry(shot_xy)
    opp = [p["xy"] for p in people if p["role"] != "teammate"]
    outfield = [p["xy"] for p in people if p["role"] == "opponent"]
    gk = next((p["xy"] for p in people if p["role"] == "gk"), None)
    return geometry(shot_xy, defenders_xy=np.array(outfield).reshape(-1, 2), opponents_xy=np.array(opp).reshape(-1, 2),
                    goalkeeper_xy=gk)


def overlay(frame_path: Path, det: pd.Series | None, out: Path) -> None:
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(frame_path)
    if det is not None:
        boxes = np.asarray(det.det_xyxy, float).reshape(-1, 4)
        for box, cls, team in zip(boxes, det.det_class, det.det_team, strict=True):
            colour = {"goalkeeper": (0, 215, 255), "referee": (40, 40, 40)}.get(
                cls, {0: (235, 120, 40), 1: (60, 60, 230)}.get(int(team), (200, 200, 200)))
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            if cls == "goalkeeper":
                cv2.putText(img, "GK", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 2)
        balls = np.asarray(det.ball_xyxy, float).reshape(-1, 4)
        for (x1, y1, x2, y2), conf in zip(balls, det.ball_conf, strict=True):
            c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(img, c, 14, (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.2f}", (c[0] + 16, c[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def pitch_plot(sb_shot: np.ndarray, people: list[dict], det: pd.Series | None, feat: pd.Series | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=110)
    ax.set_facecolor("#e9f2e6")
    x0 = 105 - 36
    for xs, ys in [([105, 105 - 16.5, 105 - 16.5, 105], [34 - 20.16, 34 - 20.16, 34 + 20.16, 34 + 20.16]),
                   ([105, 105 - 5.5, 105 - 5.5, 105], [34 - 9.16, 34 - 9.16, 34 + 9.16, 34 + 9.16])]:
        ax.plot(xs, ys, color="#7a9a74", lw=1)
    ax.plot([105, 105], [0, 68], color="#7a9a74", lw=1)
    ax.plot([105, 105], [POSTS[0][1], POSTS[1][1]], color="black", lw=4)
    colours = {"teammate": "#2b6cb0", "opponent": "#c53030", "gk": "#111111"}
    for p in people:
        ax.scatter(*p["xy"], s=46, color=colours[p["role"]], zorder=3)
    ax.scatter(*sb_shot, marker="*", s=220, color="#d69e2e", edgecolor="black", zorder=4, label="StatsBomb: strzał")
    ax.plot([sb_shot[0], POSTS[0][0]], [sb_shot[1], POSTS[0][1]], color="#d69e2e", lw=.8)
    ax.plot([sb_shot[0], POSTS[1][0]], [sb_shot[1], POSTS[1][1]], color="#d69e2e", lw=.8)
    if det is not None and feat is not None and np.isfinite(feat.attack_right):
        xy = np.asarray(det.det_pitch_xy, float).reshape(-1, 2)
        ok = np.isfinite(xy).all(axis=1)
        xy = align(xy[ok], bool(feat.attack_right)) if ok.any() else xy[ok]
        teams = np.asarray(det.det_team)[ok]
        cls = np.asarray(det.det_class)[ok]
        for (x, y), t, c in zip(xy, teams, cls, strict=True):
            edge = "#d4a017" if c == "goalkeeper" else {0: "#dd6b20", 1: "#6b46c1"}.get(int(t), "#888888")
            ax.scatter(x, y, s=70, facecolor="none", edgecolor=edge, lw=1.6, zorder=2)
        if np.isfinite([feat.shot_x, feat.shot_y]).all():
            ax.scatter(feat.shot_x, feat.shot_y, marker="*", s=220, facecolor="none", edgecolor="#d69e2e", lw=1.6,
                       zorder=4, label="detektor: strzał")
        if np.isfinite([feat.gk_x, feat.gk_y]).all():
            ax.scatter(feat.gk_x, feat.gk_y, marker="s", s=90, facecolor="none", edgecolor="#111111", lw=1.6, zorder=4,
                       label="detektor: bramkarz" + (" (proxy)" if feat.gk_proxy == 1 else ""))
    ax.set_xlim(x0, 106.5)
    ax.set_ylim(68, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=7, framealpha=.85)
    ax.set_title("pełne = StatsBomb, puste = detektor (metry, wyrównane)", fontsize=8)
    fig.text(.99, .01, "Data: StatsBomb.", ha="right", fontsize=7, color="#555555")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fmt(v, digits=1, scale=1.0) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v * scale:.{digits}f}"


def card(cat: tuple, row: pd.Series, files: dict, sb: dict, feat: pd.Series | None, residual: float) -> str:
    slug, label, _, _, finding = cat
    goal = "GOL" if row.goal_sn else "bez gola"
    probs = "".join(f"<tr><td>{name}</td><td>{fmt(row[col], 1, 100)}%</td></tr>" for col, name in METHODS)
    d = {k: (feat[k] if feat is not None else np.nan) for k in ["dist", "angle", "gk_depth", "ff_near_opp", "ff_def_cone"]}
    geo_rows = [("odległość do bramki [m]", sb["dist"], d["dist"], 1, 1), ("kąt [°]", sb["angle"], d["angle"], 0, 180 / np.pi),
                ("bramkarz od linii [m]", sb["gk_depth"], d["gk_depth"], 1, 1),
                ("najbliższy rywal [m]", sb["ff_near_opp"], d["ff_near_opp"], 1, 1),
                ("obrońcy w stożku", sb["ff_def_cone"], d["ff_def_cone"], 0, 1)]
    geo = "".join(f"<tr><td>{n}</td><td>{fmt(a, k, s)}</td><td>{fmt(b, k, s)}</td></tr>" for n, a, b, k, s in geo_rows)
    flags = "" if feat is None else (
        f"homografia {'tak' if feat.homography_ok == 1 else 'nie'} · piłka {'tak' if feat.ball_ok == 1 else 'nie'}"
        f" · bramkarz {'wykryty' if feat.gk_detected == 1 else ('proxy' if feat.gk_proxy == 1 else 'brak')}"
        f" · graczy {int(feat.n_players) if np.isfinite(feat.n_players) else '—'}")
    tags = []
    for col, name in TAGS:
        value = row[col]
        if name is None:
            tags.append(str(value))
        elif value not in ("no", "none") and isinstance(value, str):
            tags.append(name if value == "yes" else f"{name}: {value}")
    strip = "".join(f'<figure><img src="{files[o]}" loading="lazy"><figcaption>{o} ms</figcaption></figure>'
                    for o in OFFSETS)
    arrow = "obraz wyżej niż StatsBomb" if row.direction == ABOVE else "obraz niżej niż StatsBomb"
    return f"""
<section id="{slug}">
  <h2>{html.escape(label)} <span class="dir">{arrow}, różnica logitów {row.gap_F_minus_V:+.2f}</span></h2>
  <p class="finding">W raporcie: {html.escape(finding)}.</p>
  <div class="top">
    <figure class="big"><img src="{files['overlay']}"><figcaption>0 ms z detekcjami (pomarańcz/fiolet = drużyny,
      żółty = bramkarz, zielone kółko = piłka)</figcaption></figure>
    <img class="pitch" src="{files['pitch']}">
    <div class="tables">
      <p class="outcome"><b>{goal}</b> · StatsBomb: {html.escape(str(row.sb_outcome))}</p>
      <table><tr><th>metoda</th><th>xG</th></tr>{probs}</table>
      <table><tr><th></th><th>StatsBomb</th><th>detektor</th></tr>{geo}</table>
      <p class="small">{flags}<br>przesunięcie czasu SoccerNet–StatsBomb: {fmt(residual, 0)} ms</p>
    </div>
  </div>
  <div class="strip">{strip}</div>
  <p class="small">{html.escape(' · '.join(tags))}<br>{html.escape(row.match)} · {html.escape(row.key)}</p>
</section>"""


PAGE = """<!doctype html><html lang="pl"><meta charset="utf-8"><title>xg-compare: galeria rozjazdów</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;background:#fafafa;color:#1a1a1a}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 18px;margin:0 0 22px}}
h2{{margin:0 0 4px;font-size:18px}} .dir{{font-weight:400;color:#666;font-size:14px;margin-left:8px}}
.finding{{margin:2px 0 10px;color:#444}} .top{{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}}
.big img{{width:760px;max-width:100%}} .pitch{{width:360px}} .tables table{{border-collapse:collapse;margin-bottom:8px}}
td,th{{padding:2px 8px;border-bottom:1px solid #eee;text-align:left}} figure{{margin:0}}
figcaption{{font-size:12px;color:#666}} .strip{{display:flex;gap:6px;margin-top:10px}} .strip img{{width:300px}}
.small{{font-size:12px;color:#666}} .outcome{{font-size:15px;margin:0 0 6px}} nav a{{margin-right:10px}}
</style>
<h1>Największe rozjazdy obraz – StatsBomb, po jednym strzale na kategorię</h1>
<p>Z listy 5% strzałów o największej różnicy |F − V| (skala główna, 135 meczów). W każdej kategorii strzał z największą
różnicą w kierunku, w którym kategoria się myli; każdy strzał tylko raz. Materiał lokalny: nie commitować, nie publikować.
Data: StatsBomb.</p>
<nav>{nav}</nav>
{cards}
<p class="small">Data: StatsBomb. SoccerNet frames, local use only.</p>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    dis = pd.read_csv(DISAGREEMENT)
    chosen = pick(dis)
    keys = [row.key for _, row in chosen]
    table = pd.read_parquet(TABLE, columns=["key", "soccernet_shot_id", "statsbomb_event_id", "sb_x", "sb_y"]).set_index("key")
    index = pd.read_parquet(INDEX).set_index("shot_id")
    shot_ids = table.loc[keys, "soccernet_shot_id"].tolist()
    frames = pd.read_parquet(E3 / "frames.parquet", filters=[("shot_id", "in", shot_ids), ("offset_ms", "==", 0)])
    frames = frames.set_index("shot_id")
    feats = pd.read_parquet(E3 / "features.parquet", filters=[("shot_id", "in", shot_ids), ("offset_ms", "==", 0)])
    feats = feats.set_index("shot_id")
    freeze = audit.read_freeze_frames(audit.SHOTS, table.loc[keys, "statsbomb_event_id"].tolist())
    residual = pd.read_parquet(audit.PAIRS, columns=["shot_id", "residual_ms"]).set_index("shot_id").residual_ms

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "img").mkdir(parents=True)
    cards, nav = [], []
    for cat, row in chosen:
        slug = cat[0]
        sid = table.loc[row.key, "soccernet_shot_id"]
        idx = index.loc[sid]
        files = {}
        for o in OFFSETS:
            src = Path(idx["frame_path" if o == "+0" else f"frame_path_{o}"])
            dst = args.out / "img" / f"{slug}_{o}.jpg"
            shutil.copyfile(src, dst)
            files[o] = f"img/{dst.name}"
        det = frames.loc[sid] if sid in frames.index else None
        feat = feats.loc[sid] if sid in feats.index else None
        overlay(Path(idx.frame_path), det, args.out / "img" / f"{slug}_overlay.jpg")
        files["overlay"] = f"img/{slug}_overlay.jpg"
        sb_shot = statsbomb_to_metres([table.loc[row.key, "sb_x"], table.loc[row.key, "sb_y"]])
        people = statsbomb_people(freeze.get(table.loc[row.key, "statsbomb_event_id"]))
        pitch_plot(sb_shot, people, det, feat, args.out / "img" / f"{slug}_pitch.png")
        files["pitch"] = f"img/{slug}_pitch.png"
        cards.append(card(cat, row, files, statsbomb_geometry(sb_shot, people), feat, float(residual.get(sid, np.nan))))
        nav.append(f'<a href="#{slug}">{html.escape(cat[1])}</a>')
        print(f"{slug:15s} {row.direction}  gap {row.gap_F_minus_V:+.2f}  {sid}")
    (args.out / "index.html").write_text(PAGE.format(nav=" ".join(nav), cards="".join(cards)), encoding="utf-8")
    print(f"wrote {args.out / 'index.html'} ({len(cards)} cards)")


if __name__ == "__main__":
    main()
