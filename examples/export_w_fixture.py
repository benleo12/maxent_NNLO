"""Export a predeclared uniform subset of a compatible full W prior.

    python examples/export_w_fixture.py wp FULL_PRIOR.npz OUTPUT.npz

The public demonstration uses 80,000 selected events per charge. The selector is
the lowest SHA256 hashes of (fixed salt, charge, original row ID), independent of
kinematics and fit results. Original four-vectors, weights and IDs are retained.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from maxent_upgrade.provenance import canonical_json, file_hash
if __package__:
    from .w_coordinates import coordinates
else:
    from w_coordinates import coordinates


def export(charge, source, output, count=80000):
    source, output = Path(source), Path(output)
    with np.load(source, allow_pickle=False) as archive:
        weight = archive["weight"]
        fid = ((archive["pt_l"] > 20) & (np.abs(archive["y_l"]) < 2.5)
               & (archive["pt_nu"] > 25) & (archive["mt"] > 40)
               & (archive["charge"] == (1 if charge == "wp" else -1))
               & np.isfinite(weight) & (weight > 0))
        ids = np.flatnonzero(fid)
        n = min(count, len(ids))
        if n < 2:
            raise ValueError("at least two selected events are required")
        scores = [hashlib.sha256(f"w-public-v1:731131:{charge}:{i}".encode()).digest() for i in ids]
        positions = np.argsort(np.asarray(scores, dtype="S32"), kind="stable")[:n]
        selected_ids = np.sort(ids[positions])
        lepton, neutrino = archive["l_born"][selected_ids], archive["nu"][selected_ids]
        events = coordinates(lepton, neutrino)
        original_weight = weight[selected_ids]
        probability = n / len(ids)
        n_total = int(archive["n_total"])
        meta = dict(schema="maxent_upgrade.w-uniform-fixture.v1", charge=charge,
                    source_name=source.name, source_sha256=file_hash(source),
                    source_generated_events=n_total, source_selected_events=len(ids),
                    selected_events=n, selection="pt_l>20, pt_nu>25, mt>40 GeV; abs(y_l)<2.5; requested charge; positive finite original weight",
                    sampling=dict(method="lowest SHA256 of w-public-v1:731131:charge:original_row_id",
                                  allocation="uniform fixed count; no kinematic strata or fit-dependent selection",
                                  inclusion_probability=probability,
                                  interpretation="compact demonstration; independent-row errors refer to the unconditional generator sample, not conditional resampling of this fixed parent pool"),
                    four_vector_order=["px", "py", "pz", "E"], energy_units="GeV",
                    coordinate_convention="massless Cartesian Collins–Soper; charged lepton for both charges; z/y signed by boson pz",
                    zero_recoil="sx,sy set to zero when qT<=1e-12; rho_sx,rho_sy are continuous functions",
                    source_selected_cross_section_pb=float(weight[fid].sum() / n_total),
                    subset_estimated_cross_section_pb=float(original_weight.sum() / probability / n_total),
                    generated_flags=archive["flags"].tolist() if "flags" in archive else None)
        events.update(l_born=lepton, nu=neutrino, event_id=selected_ids, source_index=selected_ids,
                      original_weight=original_weight, inclusion_probability=np.full(n, probability),
                      inverse_inclusion_probability=np.full(n, 1 / probability),
                      weight=original_weight / probability, charge=np.full(n, 1 if charge == "wp" else -1),
                      n_total=np.array(n_total))
    qt_edges, c_edges, y_edges = [0, 10, 25, 50, 100, 500, np.inf], [0, .6, .7, .8, 1.00001], [0, 1, 2, 4, np.inf]
    points = np.column_stack([events["pt_w"], np.abs(events["c"]), events["y_w"].__abs__()])
    bins = (qt_edges, c_edges, y_edges)
    counts = np.histogramdd(points, bins=bins)[0]
    totals = np.histogramdd(points, bins=bins, weights=events["weight"])[0]
    squares = np.histogramdd(points, bins=bins, weights=events["weight"]**2)[0]
    effective = np.divide(totals**2, squares, out=np.zeros_like(totals), where=squares > 0)
    meta["local_statistics"] = dict(axis_names=["qT_GeV", "abs_cos_theta", "abs_y"],
                                    edges=[[None if np.isinf(v) else v for v in a] for a in bins],
                                    counts=counts.astype(int).tolist(), effective_events=effective.tolist(),
                                    interpretation="descriptive populated-cell statistics, not a support certificate")
    events["metadata"] = np.array(canonical_json(meta))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **events)
    meta["fixture_sha256"] = file_hash(output)
    output.with_suffix(".json").write_text(canonical_json(meta) + "\n")
    print(f"{charge}: selected {n:,} of {len(ids):,}; {output.stat().st_size / 1e6:.2f} MB; {meta['fixture_sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("charge", choices=("wp", "wm"))
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    export(args.charge, args.source, args.output)
