"""Measured trade-off on MT-Bench human judgments (150 pairs: 50 first / 50 second / 50 tie; random = 0.33; CC BY 4.0, from the Kodiak
repo's real-anchor eval). For each threshold: the share of pairs the judge accepts and its accuracy on them, for three policies:
one order only, both orders averaged, and both orders + escalate when the orders disagree. This is an eval set: it is used to report,
never to train or to pick a model.

    uv run python scripts/curve.py --model ../kodiak/dist/kodiak-v0.4-1b --out reports/curve-v0.4-1b.json
"""
import argparse
import json
from pathlib import Path

from kodiak_judge.judge import Judge

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="../kodiak/dist/kodiak-v0.4-1b")
ap.add_argument("--eval", default="../kodiak/data/eval/kodiak-real-anchors-v0.1.jsonl")
ap.add_argument("--out", required=True)
a = ap.parse_args()
items = [json.loads(line) for line in open(a.eval)]
items = [e for e in items if "anchor:mtbench" in e["meta"]["tags"]]
gold = [e["answers"]["decision"]["label"] for e in items]
j = Judge.load(a.model)
both = j.judge_many([e["state"] for e in items], threshold=0.0)
j1 = Judge(j.kodiak, both_orders=False)
one = j1.judge_many([e["state"] for e in items], threshold=0.0)
policies = {
    "one order": [(r["confidence"], r["verdict"] == g, True) for r, g in zip(one, gold)],
    "both orders averaged": [(r["confidence"], r["verdict"] == g, True) for r, g in zip(both, gold)],
    "both orders + escalate on disagreement": [(r["confidence"], r["verdict"] == g, not r["orders"]["disagree"]) for r, g in zip(both, gold)],
}
report = {"model": a.model, "eval": "MT-Bench human judgments, 150 pairs (random = 0.33)",
          "orders_disagree_share": round(sum(r["orders"]["disagree"] for r in both) / len(both), 3), "policies": {}}
for name, rows in policies.items():
    report["policies"][name] = {"accuracy_all": round(sum(ok for _, ok, _ in rows) / len(rows), 3), "curve": []}
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        acc = [ok for c, ok, keep in rows if c >= t and keep]
        report["policies"][name]["curve"].append({"threshold": t, "accepted_share": round(len(acc) / len(rows), 3),
                                                  "accuracy_when_accepted": round(sum(acc) / len(acc), 3) if acc else None})
Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
print("orders disagree on", report["orders_disagree_share"], "of pairs")
for name, p in report["policies"].items():
    print(f"{name}: all {p['accuracy_all']}; " + "; ".join(f"t{c['threshold']}: {c['accepted_share']:.0%} at {c['accuracy_when_accepted']}" for c in p["curve"]))
