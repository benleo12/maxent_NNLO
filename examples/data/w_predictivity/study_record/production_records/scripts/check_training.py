"""Review frozen training-only decisions and all prescribed candidate outcomes."""
from pathlib import Path
import argparse
import json
import hashlib

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--runs", type=int, required=True)
a = ap.parse_args()
base = a.root / ("fits%d" % a.runs)
result = dict(runs_per_channel=a.runs, validation_opened=False, charges={})
for charge in ("WP", "WM"):
    selections = {}
    for subset in ("full", "even", "odd"):
        path = base / charge / subset / "SELECTION.json"
        selections[subset] = json.loads(path.read_text())
    decisions = {key: [(step["block"], step["decision"]) for step in rec["history"]]
                 for key, rec in selections.items()}
    stable = (all(rec["status"] == "training_sequence_complete" for rec in selections.values())
              and len({rec["selected"] for rec in selections.values()}) == 1
              and all(value == decisions["full"] for value in decisions.values()))
    controls = {}
    for path in sorted((base / charge / "full").glob("candidate_*.json")):
        record = json.loads(path.read_text())
        controls[record["name"]] = dict(checks=record["checks"],
                                        ess_retained=record["fit"]["ess_retained"],
                                        q_over_p_range=record["fit"]["q_over_p_range"])
    if len(controls) != 18:
        raise ValueError("all18 prescribed controls must have a saved outcome before review")
    result["charges"][charge] = dict(stable=stable,
        selected={key: rec["selected"] for key, rec in selections.items()},
        status={key: rec["status"] for key, rec in selections.items()}, decisions=decisions,
        controls=controls, needs_extension=not stable and a.runs < 640,
        final_status="stable" if stable else ("inconclusive_at_cap" if a.runs >= 640 else "extend_training_only"))
result["ready_to_freeze"] = all(rec["stable"] or a.runs >= 640 for rec in result["charges"].values())
result["maximum_training_runs_per_channel"] = 640
target = base / "TRAINING_REVIEW.json"
assert not target.exists()
target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, indent=2))
