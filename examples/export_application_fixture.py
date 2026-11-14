"""Uniform compact samples of existing Higgs, diphoton or Z+jet event arrays.

    python examples/export_application_fixture.py PROCESS OUTPUT SOURCE [SOURCE ...]

The selection uses 80,000 Higgs/diphoton or 200,000 Z+jet uniformly sampled valid
rows with seed731132 and source order sorted by basename. Source IDs, raw relative weights,
normalization factors and inverse-inclusion factors remain explicit.
"""
import argparse
from pathlib import Path

import numpy as np

from maxent_upgrade.provenance import canonical_json, file_hash


def valid_rows(process, source):
    weight = source["weight"]
    valid = np.isfinite(weight) & (weight > 0)
    if process == "zj":
        valid &= source["njet"] >= 1
        valid &= np.isfinite(source["mll"]) & (source["mll"] > 0)
    elif process == "aa":
        valid &= np.isfinite(source["m_aa"]) & (source["m_aa"] > 0)
    else:
        valid &= np.isfinite(source["pT_H"]) & np.isfinite(source["y_H"])
    return np.flatnonzero(valid)


def extract(process, source, indices):
    def get(key):
        return np.asarray(source[key])[indices]
    if process == "zj":
        keys = ("mll", "y_abs", "ptj1", "ptj2", "yj1", "yj2", "njet", "dphi_ll", "pimdphi", "gidx")
        return {k: get(k) for k in keys if k in source}
    if process == "ggh":
        out = {k: get(k) for k in ("pT_H", "pT_j1", "pT_j1_20", "n_j30", "fid", "pT_yy", "y_yy") if k in source}
        out["y_abs"] = np.abs(get("y_H"))
        out["y_H"] = get("y_H")
        return out
    g1, g2 = get("g1"), get("g2")
    mass, pt = get("m_aa"), get("pt_aa")
    numerator = ((g1[:, 3] + g1[:, 2]) * (g2[:, 3] - g2[:, 2])
                 - (g1[:, 3] - g1[:, 2]) * (g2[:, 3] + g2[:, 2]))
    angle = np.clip(np.abs(numerator) / np.maximum(mass * np.hypot(mass, pt), 1e-12), 0, 1)
    pt1, pt2 = np.hypot(g1[:, 0], g1[:, 1]), np.hypot(g2[:, 0], g2[:, 1])
    return dict(g1=g1, g2=g2, m_aa=mass, pt_aa=pt, y_abs=np.abs(get("y_aa")),
                abs_costh_aa=angle, dphi_aa=np.abs(get("dphi_aa")),
                pimdphi=np.pi - np.abs(get("dphi_aa")),
                pt_lead=np.maximum(pt1, pt2), pt_sub=np.minimum(pt1, pt2))


def export(process, output, paths, count=None):
    count = (200000 if process == "zj" else 80000) if count is None else count
    if count < 2:
        raise ValueError("request at least two events")
    paths = sorted(map(Path, paths), key=lambda p: p.name)
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("source basenames must be unique")
    rows, records = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            indices = valid_rows(process, source)
            normalization = float(source["sigma_pb"]) / float(source["n_generated"]) if process == "aa" else 1.
            if not np.isfinite(normalization) or normalization <= 0:
                raise ValueError("source normalization must be finite and positive")
            record = dict(name=path.name, sha256=file_hash(path), stored_events=len(source["weight"]),
                          valid_events=len(indices), normalization=normalization)
            for key in ("n_generated", "sigma_pb", "n_total", "sigma_gen_mb", "tms", "pthatmin", "pthatmax"):
                if key in source:
                    record[key] = source[key].tolist()
            rows.append(indices)
            records.append(record)
    total = sum(len(ids) for ids in rows)
    chosen = np.sort(np.random.default_rng(731132).choice(total, min(count, total), replace=False))
    probability = len(chosen) / total
    pieces, offset = [], 0
    for file_index, (path, indices, record) in enumerate(zip(paths, rows, records)):
        local = chosen[(chosen >= offset) & (chosen < offset + len(indices))] - offset
        original_ids = indices[local]
        offset += len(indices)
        record["fixture_events"] = len(local)
        if not len(local):
            continue
        with np.load(path, allow_pickle=False) as source:
            piece = extract(process, source, original_ids)
            original_weight = source["weight"][original_ids]
            piece.update(original_weight=original_weight,
                         source_normalization=np.full(len(local), record["normalization"]),
                         source_file_index=np.full(len(local), file_index, dtype=np.int16),
                         source_row=original_ids,
                         event_id=(np.uint64(file_index) << np.uint64(48)) + original_ids.astype(np.uint64),
                         weight=original_weight * record["normalization"] / probability)
            pieces.append(piece)
    if not pieces or any(set(p) != set(pieces[0]) for p in pieces):
        raise ValueError("all nonempty source files must provide the same analysis fields")
    arrays = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    arrays["inclusion_probability"] = np.full(len(chosen), probability)
    arrays["inverse_inclusion_probability"] = np.full(len(chosen), 1 / probability)
    meta = dict(schema="maxent_upgrade.uniform-application-fixture.v1", process=process,
                events=len(chosen), parent_valid_events=total, sources=records,
                sampling=dict(method="uniform without replacement over concatenated valid rows",
                              seed=731132, source_order="basename", requested_events=count,
                              inclusion_probability=probability,
                              frozen_before_fitting=True),
                weight_definition="original_weight * source_normalization / inclusion_probability",
                normalization_note=("source_normalization=sigma_pb/n_generated for each exclusive/merged diphoton source"
                                    if process == "aa" else "original relative generator weights retained; target cross section sets posterior pb weights"),
                selection=("stored diphoton fiducial/isolation selection, finite positive mass and weight"
                           if process == "aa" else "stored DY lepton selection, njet>=1, finite positive mass and weight"
                           if process == "zj" else "inclusive on-shell Higgs, finite kinematics and positive weight"),
                interpretation="compact executable demonstration; distinct event statistics from full paper fits")
    arrays["metadata"] = np.array(canonical_json(meta))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    meta["fixture_sha256"] = file_hash(output)
    output.with_suffix(".json").write_text(canonical_json(meta) + "\n")
    print(f"{process}: {len(chosen):,}/{total:,} events, {output.stat().st_size / 1e6:.2f} MB, {meta['fixture_sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("process", choices=("ggh", "aa", "zj"))
    parser.add_argument("output")
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    export(args.process, args.output, args.sources, count=args.count)
