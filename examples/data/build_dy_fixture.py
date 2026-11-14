"""Build a compact, stratified subset of the paper's current DY prior.

The sampling rule below was fixed before fitting this subset. Run on the
machine holding the production sample, not by loading it on a working laptop.
The released fixture retains original source IDs and sampling probabilities.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEED = 13112026
EDGES = [0., 30., 60., 100., 200., 500., np.inf]
LIMITS = [50000, 20000, 20000, 15000, 12000, 12000]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior")
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    rng = np.random.default_rng(SEED)
    with np.load(args.prior) as source:
        recoil = source["pT_ll"]
        selected, probabilities, strata = [], [], []
        for lo, hi, limit in zip(EDGES[:-1], EDGES[1:], LIMITS):
            available = np.flatnonzero((recoil >= lo) & (recoil < hi))
            count = min(limit, len(available))
            chosen = rng.choice(available, count, replace=False) if count else available
            selected.extend(chosen)
            probability = count / len(available) if len(available) else 0
            probabilities.extend([probability] * count)
            strata.append(dict(low=lo, high=hi if np.isfinite(hi) else None,
                               population=len(available), selected=count, inclusion_probability=probability))
        order = np.argsort(selected)
        idx = np.asarray(selected, dtype=np.int64)[order]
        inclusion = np.asarray(probabilities)[order]
        original = source["w"][idx]
        if not np.isfinite(original).all() or not (original > 0).all():
            raise ValueError("source weights must be finite and strictly positive")
        fields = dict(event_id=source["gidx"][idx], source_index=idx,
                      mll=source["mll"][idx], y_abs=np.abs(source["y_ll"][idx]),
                      pT_ll=recoil[idx], abs_costh=np.abs(source["costh_cs"][idx]),
                      phistar=source["phistar"][idx], pT_lead=source["pT_lead"][idx],
                      pT_sub=source["pT_sub"][idx], ptj1=source["ptj1_had"][idx],
                      jet_count=source["njet_had"][idx],
                      weight=original / inclusion, original_weight=original,
                      inclusion_probability=inclusion)
        if len(np.unique(fields["event_id"])) != len(idx):
            raise ValueError("source IDs are not unique")
        record = dict(source_file=Path(args.prior).name, source_sha256=digest(args.prior),
                      source_events=len(recoil), seed=SEED, sampling="uniform without replacement within recoil strata",
                      strata=strata, weight="original_weight / inclusion_probability",
                      event_id="source gidx, unchanged", source_index="row in the full selected prior")
    fields["fixture_provenance_json"] = np.asarray(json.dumps(record, sort_keys=True))
    np.savez_compressed(output, **fields)
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {len(idx):,} events, {output.stat().st_size / 1e6:.2f} MB; SHA256 {digest(output)}")


if __name__ == "__main__":
    main()
