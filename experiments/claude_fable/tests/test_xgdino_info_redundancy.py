"""The ticket 13 read-out (scripts/xgdino_info_redundancy.py) on a synthetic redundancy run: the tables carry the
contract's statistics and verdicts, the leak rows join the estimands, subgroup counts come from the keys, the leaf
comparison finds differences, and the run check refuses unpinned or edited runs."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xgdino_info_redundancy as script  # noqa: E402
from test_xgdino_info_protocol import DRAWS, population, subsets  # noqa: E402
from xgpoc.bundle import sha256_file  # noqa: E402
from xgpoc.dino import info_protocol as ip  # noqa: E402
from xgpoc.stack import stats as ss  # noqa: E402


@pytest.fixture(scope="module")
def run():
    table, attributes, y = population()
    design, columns = ip.design_table(table, attributes)
    prepared = ip.prepare(design, columns, subsets(table), blocks=("Xc", *ip.FAMILY), draws=DRAWS)
    p, _ = ip.fit_arms(prepared, y)
    redundancy = ip.redundancy(y, p, prepared.block.weights, prepared.masks, not_explain_available=False)
    return json.loads(json.dumps(redundancy, default=float))


def test_verdict_table_carries_the_contract(run):
    t = script.verdict_table(run).set_index("block")
    assert list(t.index) == ["Xc", *ip.FAMILY, "Q"]
    for b, r in run["blocks"].items():
        assert t.loc[b, "gain_delta"] == r["gain"]["delta"] and t.loc[b, "R_delta"] == r["R"]["delta"]
        assert t.loc[b, "R_over_G"] == pytest.approx(r["R"]["delta"] / r["G"]["delta"])
        assert (t.loc[b, "raw_verdict"], t.loc[b, "verdict"]) == (r["raw_verdict"], r["verdict"])
    assert t.loc["crowding", "verdict"] in ("explains most", "explains part")
    gated = t[t.raw_verdict == "does not explain"]
    assert len(gated) and (gated.verdict == "inconclusive").all()


def test_estimand_and_leak_tables_join_the_restricted_reads(run):
    e = script.estimand_table(run)
    assert set(e.estimand) == {"no_v3", "unseen", "unseen_standardised"} and len(e) == 3 * len(run["blocks"])
    leak = script.leak_table(run).set_index(["block", "statistic"])
    assert len(leak) == len(run["blocks"]) * len(ip.LEAK_STATISTICS)
    row = leak.loc[("Xc", "R")]
    assert row.no_v3_delta == run["estimands"]["no_v3"]["Xc"]["R"]["delta"]
    assert row["seen-unseen_delta"] == run["leak"]["Xc"]["R"]["heterogeneity"]["seen-unseen"]["delta"]
    assert row.leak_verdict == run["leak"]["Xc"]["R"]["verdict"]


def test_subgroup_table_takes_counts_from_the_keys():
    rng = np.random.default_rng(1)
    n = 3000
    idx = pd.Index([f"s{i}" for i in range(n)])
    y = (rng.random(n) < 0.15).astype(int)
    inside = rng.random(n) < 0.3
    keys = pd.DataFrame({"set_piece": ip.side(inside, np.ones(n, bool)),
                         "tiny": ip.side(np.arange(n) < 50, np.ones(n, bool))}, index=idx)
    out = ip.subgroups(y, rng.normal(0, 0.01, n), ss.bootstrap_weights(np.repeat(np.arange(100), 30), DRAWS, 0),
                       keys, pd.Series(dtype=float), 0.15)
    counts = pd.DataFrame([{"key": k, "side": s, "shots": int((keys[k] == s).sum()),
                            "goal_count": int(y[(keys[k] == s).to_numpy()].sum())} for k in keys for s in ("in", "out")])
    t = script.subgroup_table(out, counts).set_index(["key", "side"])
    assert t.loc[("set_piece", "in"), "goal_count"] == y[inside].sum()
    assert t.loc[("set_piece", "in"), "mean_d_goals"] == out["keys"]["set_piece"]["sides"]["in"]["goals"]
    assert not t.loc[("tiny", "in"), "readable"]
    with pytest.raises(SystemExit):
        script.subgroup_table(out, counts.assign(shots=counts.shots + 1))


def test_leaves_reports_the_largest_difference_and_mismatches():
    a = {"x": 1.0, "y": [0.5, float("nan")], "v": "inconclusive"}
    assert script.leaves(a, a) == (4, 0.0, [])
    n, worst, bad = script.leaves(a, {"x": 1.25, "y": [0.5, float("nan")], "v": "explains part"})
    assert (n, worst) == (4, 0.25) and bad == ["/v: 'inconclusive' != 'explains part'"]
    assert script.leaves(a, {"x": 1.0})[2] == [": keys differ"]


def test_checked_run_refuses_unpinned_or_edited_runs(tmp_path):
    folder, storage = tmp_path / "real", tmp_path / "storage"
    folder.mkdir(), storage.mkdir()
    (folder / "summary.json").write_text("{}")
    (storage / "probabilities.csv").write_text("shot_id\n")
    record = {"mode": "real", "protocol_sha256": script.run.PROTOCOL_SHA256, "pinned": script.run.PROTOCOL_SHA256,
              "files": {"summary.json": sha256_file(folder / "summary.json")},
              "storage": {"root": str(storage), "probabilities.csv": sha256_file(storage / "probabilities.csv")}}
    (folder / "complete.json").write_text(json.dumps(record))
    assert script.checked_run(folder)["mode"] == "real"
    (folder / "complete.json").write_text(json.dumps(record | {"protocol_sha256": "0" * 64}))
    with pytest.raises(SystemExit):
        script.checked_run(folder)
    (folder / "complete.json").write_text(json.dumps(record))
    (folder / "summary.json").write_text('{"edited": 1}')
    with pytest.raises(SystemExit):
        script.checked_run(folder)
