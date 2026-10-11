"""Table shells of the xg-vendor pre-registration (ticket 14) and their filling from the analysis outputs (tickets 14, 15).

`render(params)` gives the empty shells exactly as the pre-registration document prints them: fixed rows and columns,
empty number cells, the pre-registered thresholds and roles already in. `render(params, Outputs(...), margin)` fills the
same shells from the folders `scripts/vendor_ladder.py`, `vendor_probes.py` and `vendor_leakage.py` write. The reading
columns apply the document's decision rules (`h2_reading`, `h3_reading`, `difference_reading`). Shells S2-S4 repeat
for each population set; a set the run skipped is printed as such.

Formats: log-loss, Brier, differences in log-loss and margins 4 decimals; AUC, R2, slopes and coefficients 3; p-values 4;
intervals are 95% percentile match-bootstrap intervals "[lo, hi]" unless the column says 90%.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xgpoc.vendor import ladder as ld
from xgpoc.vendor import leakage as lk
from xgpoc.vendor.probe_targets import BINARY_TARGETS, TARGET_COLUMNS
from xgpoc.vendor.probes import CONTROL

ALPHA = ld.ALPHA
POPULATIONS = {"primary": "pierwotna", "combined": "łączona"}
RUNG_NAMES = {"constant": "stała", "dist_angle": "odległość i kąt", "event": "zdarzenia", "freeze_frame": "freeze frame (pozycje)",
              "vendor": "xG dostawcy", "frame": "klatka (G-combo)", "frame+positions": "klatka + pozycje", "frame+vendor": "klatka + dostawca"}
METRIC_COLUMNS = (("log_loss", 4, "Log-loss [95%]"), ("brier", 4, "Brier [95%]"), ("auc", 3, "AUC [95%]"),
                  ("cal_slope", 3, "Nachylenie kalibracji [95%]"), ("cal_intercept", 3, "Wyraz wolny kalibracji [95%]"),
                  ("mean_p_minus_rate", 4, "Śr. p − częstość [95%]"))
KEY_CONTRASTS = (("frame", "vendor"), ("frame", "freeze_frame"), ("vendor", "freeze_frame"), ("freeze_frame", "event"),
                 ("event", "dist_angle"), ("dist_angle", "constant"), ("frame+positions", "frame"), ("frame+vendor", "frame"))
H2_ROWS = (("h2", "primary", "konfirmacyjny"), ("h2", "combined", "odporność"),
           ("frame_plus_vendor", "primary", "opisowy"), ("frame_plus_vendor", "combined", "opisowy"))
H2_NAMES = {"h2": "klatka + pozycje − pozycje", "frame_plus_vendor": "klatka + dostawca − dostawca"}
PROBE_ENCODERS = {"emb_g_fold0": "S6", "emb_g_fold1": "S7"}
H4_ROWS = tuple((spec, regressor) for spec in lk.SPECS for regressor in (spec.regressors or ("intercept",)))
H4_PARTS = {"on_target": "celność", "offset": "kontrast offsetów", "residual": "reszta czasowa"}
REGRESSOR_NAMES = {"on_target": "celny (SN)", "intercept": "średnia G(0) − G(−400)", "residual_s": "reszta [s]",
                   "early": "kotwica > 40 ms przed SB", "late": "kotwica > 40 ms po SB"}


@dataclass(frozen=True)
class Outputs:
    """Folders of one run's analysis outputs."""

    ladder: Path
    probes: Path
    leakage: Path


# ------------------------------------------------------------------ formatting
def _number(value, digits: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:+.{digits}f}"


def _plain(value, digits: int) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def _interval(value, lo, hi, digits: int, signed: bool = False) -> str:
    fmt = _number if signed else _plain
    return f"{fmt(value, digits)} [{fmt(lo, digits)}, {fmt(hi, digits)}]"


def markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join([*lines, *("| " + " | ".join(row) + " |" for row in rows)])


# ------------------------------------------------------------------ decision rules
def h2_reading(record: dict | None) -> str:
    """The pre-registered reading of an H2-type contrast (a − b in log-loss; negative = the frame helps)."""
    if record is None:
        return "pominięty"
    equivalent, superior, worse = record["p_tost"] <= ALPHA, record["p_superiority"] <= ALPHA, record["lo95"] > 0
    if superior and equivalent:
        return "poprawa istotna, mniejsza niż δ"
    if superior:
        return "klatka dodaje; równoważności przy δ brak"
    if equivalent and worse:
        return "równoważne przy δ; stos istotnie gorszy"
    if equivalent:
        return "równoważne przy δ: klatka nie dodaje więcej niż δ"
    if worse:
        return "stos istotnie gorszy; równoważności przy δ brak"
    return "nierozstrzygnięty"


def h3_reading(value: float, lo: float, hi: float, threshold: float | None, control: bool = False) -> str:
    """Threshold reading of a probe: the lower end of the 95% interval decides "koduje"; the upper end below the threshold
    gives "nie koduje"; the negative control must have R2 <= 0 (point estimate)."""
    if control:
        return "kontrola przechodzi" if value <= 0 else "kontrola NIE przechodzi"
    if lo >= threshold:
        return "koduje"
    if hi < threshold:
        return "nie koduje (poniżej progu)"
    return "nierozstrzygnięty"


def difference_reading(lo: float, hi: float) -> str:
    """Reading of G-combo minus frozen DINOv2 on the same rows."""
    if lo > 0:
        return "G > DINOv2"
    if hi < 0:
        return "G < DINOv2"
    return "brak rozróżnialnej różnicy"


# ------------------------------------------------------------------ shells
def _population_rows(ladder: dict | None) -> list[list[str]]:
    rows = []
    for name, label in POPULATIONS.items():
        s = None if ladder is None else ladder["h2"]["sets"].get(name)
        if s is None:
            rows.append([label, *([""] * 6 if ladder is None else ["pominięta"] + ["—"] * 5)])
            continue
        e = s["exclusions"]
        rows.append([label, str(e["chosen"]), str(e["penalty"]), str(e["no_frame"]), str(e["analysed"]), str(s["goals"]), str(s["matches"])])
    return rows


def _metric_rows(ladder: dict | None, population: str, variant: str) -> list[list[str]]:
    names = list(ld.LADDER) if variant == "recalibrated" else list(ld.SINGLE_RUNGS)
    metrics = None if ladder is None else ladder["metrics"]
    rows = []
    for rung in names:
        if metrics is None:
            rows.append([RUNG_NAMES[rung], *[""] * len(METRIC_COLUMNS)])
            continue
        hit = metrics[metrics.population.eq(population) & metrics.variant.eq(variant) & metrics.rung.eq(rung)]
        if hit.empty:
            rows.append([RUNG_NAMES[rung], *["pominięta"] * len(METRIC_COLUMNS)])
            continue
        r = hit.iloc[0]
        rows.append([RUNG_NAMES[rung], *(_interval(r[m], r[f"{m}_lo"], r[f"{m}_hi"], d, signed=m in ("cal_intercept", "mean_p_minus_rate"))
                                         for m, d, _ in METRIC_COLUMNS)])
    return rows


def _contrast_rows(ladder: dict | None, population: str) -> list[list[str]]:
    rows = []
    for a, b in KEY_CONTRASTS:
        label = f"{RUNG_NAMES[a]} − {RUNG_NAMES[b]}"
        if ladder is None:
            rows.append([label, "", "", "", ""])
            continue
        c = ladder["contrasts"]
        pick = c.population.eq(population) & c.variant_a.eq("recalibrated") & c.a.eq(a) & c.b.eq(b)
        ll, auc = c[pick & c.metric.eq("log_loss")], c[pick & c.metric.eq("auc")]
        if ll.empty:
            rows.append([label, "pominięty", "—", "—", "—"])
            continue
        ll, auc = ll.iloc[0], auc.iloc[0]
        rows.append([label, _interval(ll["diff"], ll.lo95, ll.hi95, 4, True), f"[{_number(ll.lo90, 4)}, {_number(ll.hi90, 4)}]",
                     _plain(ll.smallest_equivalence_margin, 4), _interval(auc["diff"], auc.lo95, auc.hi95, 3, True)])
    return rows


def _h2_rows(ladder: dict | None) -> list[list[str]]:
    rows = []
    for key, population, role in H2_ROWS:
        head = [H2_NAMES[key], POPULATIONS[population], role]
        if ladder is None:
            rows.append([*head, *[""] * 8])
            continue
        s = ladder["h2"]["sets"].get(population)
        if s is None:
            rows.append([*head, *["—"] * 7, "pominięty"])
            continue
        r = s[key]
        rows.append([*head, _plain(r["log_loss_a"], 4), _plain(r["log_loss_b"], 4), _interval(r["diff"], r["lo95"], r["hi95"], 4, True),
                     f"[{_number(r['lo90'], 4)}, {_number(r['hi90'], 4)}]", _plain(r["p_tost"], 4), _plain(r["p_superiority"], 4),
                     _plain(r["smallest_equivalence_margin"], 4), h2_reading(r)])
    return rows


def _probe_rows(probes: dict | None, encoder: str, params: dict) -> list[list[str]]:
    thresholds, confirmatory = params["h3_thresholds"], set(params["h3_confirmatory_targets"])
    rows = []
    for target in (CONTROL, *TARGET_COLUMNS):
        control = target == CONTROL
        metric = "AUC" if target in BINARY_TARGETS else "R2"
        threshold = "R2 ≤ 0" if control else f"{thresholds[target]:.2f}"
        role = "kontrola negatywna" if control else ("konfirmacyjny" if target in confirmatory else "opisowy")
        head = [target, metric]
        if probes is None:
            rows.append([*head, "", "", "", "", threshold, role, "", ""])
            continue
        m, d = probes["metrics"], probes["differences"]
        own = m[m.encoder.eq(encoder) & m.subset.eq("all") & m.coverage.eq("own") & m.target.eq(target)]
        diff = d[d.encoder.eq(encoder) & d.subset.eq("all") & d.target.eq(target)]
        if own.empty or diff.empty:
            rows.append([*head, "—", "—", "—", "—", threshold, role, "pominięty", "—"])
            continue
        g, diff = own.iloc[0], diff.iloc[0]
        ref = m[m.encoder.eq("emb_dinov2") & m.subset.eq("all") & m.target.eq(target)
                & m.coverage.isin([f"shared_with_{encoder}", "own"])].sort_values("coverage", ascending=False).iloc[0]
        if ref.n != g.n:
            raise ValueError(f"{target}: the reference row covers {ref.n} shots, the encoder {g.n}")
        rows.append([*head, str(int(g.n)), _interval(g.value, g.lo, g.hi, 3), _interval(ref.value, ref.lo, ref.hi, 3),
                     _interval(diff.difference, diff.lo, diff.hi, 3, True), threshold, role,
                     h3_reading(g.value, g.lo, g.hi, None if control else thresholds[target], control),
                     "—" if control else difference_reading(diff.lo, diff.hi)])
    return rows


def _h4_rows(leakage: dict | None) -> list[list[str]]:
    rows = []
    for spec, regressor in H4_ROWS:
        restriction = "reszta w ±200 ms" if spec.restriction != "none" else "brak"
        head = [H4_PARTS[spec.part], spec.name, REGRESSOR_NAMES[regressor], "tak" if spec.controls else "nie", restriction]
        if leakage is None:
            rows.append([*head, "", ""])
            continue
        c = leakage["coefficients"]
        hit = c[c.spec.eq(spec.name) & c.regressor.eq(regressor) & c.subset.eq("all")] if len(c) else c
        if hit.empty:
            rows.append([*head, "—", "nie uruchomiono" if spec.part == "residual" else "—"])
            continue
        r = hit.iloc[0]
        rows.append([*head, str(int(r.n)), _interval(r.coef, r.lo, r.hi, 3, True)])
    return rows


def _read(outputs: Outputs | None) -> tuple[dict | None, dict | None, dict | None]:
    if outputs is None:
        return None, None, None
    ladder = {"metrics": pd.read_csv(outputs.ladder / "metrics.csv"), "contrasts": pd.read_csv(outputs.ladder / "contrasts.csv"),
              "h2": json.loads((outputs.ladder / "h2.json").read_text())}
    probes = {"metrics": pd.read_csv(outputs.probes / "probes.csv"), "differences": pd.read_csv(outputs.probes / "probe_differences.csv")}
    leakage = {"coefficients": pd.read_csv(outputs.leakage / "leakage.csv")}
    return ladder, probes, leakage


SHELL_TITLES = {
    "S1": "S1. Populacje analizy H1/H2: liczności i wykluczenia",
    "S2": "S2. H1, drabina informacji, warianty rekalibrowane (populacja: {population})",
    "S3": "S3. H1, drabina informacji, warianty surowe (populacja: {population})",
    "S4": "S4. H1, kontrasty kluczowe, warianty rekalibrowane (populacja: {population})",
    "S5": "S5. H2 i klatka + dostawca{margin}",
    "S6": "S6. H3, sondy: G-combo fold 0 seed 0 (model rozstrzygający) wobec zamrożonego DINOv2",
    "S7": "S7. H3, wrażliwość: G-combo fold 1 seed 0 wobec zamrożonego DINOv2",
    "S8": "S8. H4, diagnostyka przecieku (opisowa)",
}
HEADERS = {
    "S1": ["Populacja", "Wybrane", "Karne", "Bez klatki", "Analizowane", "Gole", "Mecze"],
    "S2": ["Szczebel", *(h for _, _, h in METRIC_COLUMNS)],
    "S4": ["Kontrast (A − B)", "Δ log-loss [95%]", "Δ log-loss, 90%", "Najmniejszy margines równoważności", "Δ AUC [95%]"],
    "S5": ["Kontrast", "Populacja", "Rola", "Log-loss A", "Log-loss B", "Δ [95%]", "Δ, 90%", "p TOST przy δ", "p przewagi",
           "Najmniejszy margines", "Odczyt"],
    "S6": ["Cel", "Metryka", "n", "G-combo [95%]", "DINOv2 [95%]", "G − DINOv2 [95%]", "Próg", "Rola", "Odczyt progu", "Odczyt różnicy"],
    "S8": ["Część", "Specyfikacja", "Regresor", "Kontrole", "Ograniczenie", "n", "Współczynnik [95%]"],
}
HEADERS["S3"], HEADERS["S7"] = HEADERS["S2"], HEADERS["S6"]


def shell_bodies(params: dict, outputs: Outputs | None = None) -> dict[tuple[str, str | None], str]:
    """Markdown body of every shell, keyed by (shell id, population set or None)."""
    ladder, probes, leakage = _read(outputs)
    bodies = {("S1", None): markdown(HEADERS["S1"], _population_rows(ladder))}
    for population in POPULATIONS:
        bodies["S2", population] = markdown(HEADERS["S2"], _metric_rows(ladder, population, "recalibrated"))
        bodies["S3", population] = markdown(HEADERS["S3"], _metric_rows(ladder, population, "raw"))
        bodies["S4", population] = markdown(HEADERS["S4"], _contrast_rows(ladder, population))
    bodies["S5", None] = markdown(HEADERS["S5"], _h2_rows(ladder))
    for encoder, shell in PROBE_ENCODERS.items():
        bodies[shell, None] = markdown(HEADERS[shell], _probe_rows(probes, encoder, params))
    bodies["S8", None] = markdown(HEADERS["S8"], _h4_rows(leakage))
    return bodies


def render(params: dict, outputs: Outputs | None = None, margin: float | None = None) -> str:
    """All shells as one markdown text, titles included; empty when `outputs` is None."""
    parts = []
    for (shell, population), body in shell_bodies(params, outputs).items():
        title = SHELL_TITLES[shell].format(population=POPULATIONS.get(population, ""), margin="" if margin is None else f" (δ = {margin:g})")
        parts.append(f"#### {title}\n\n{body}\n")
    return "\n".join(parts)
