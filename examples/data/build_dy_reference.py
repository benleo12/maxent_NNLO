"""Export the independent angular reference and the recoil closure histograms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("costh_directory")
    parser.add_argument("recoil_cache")
    parser.add_argument("output")
    args = parser.parse_args()
    base, output = Path(args.costh_directory), Path(args.output)
    if output.exists():
        parser.error("output already exists")
    channels = ["LO", "V", "R", "VV", "RV", "RR"]
    files = {}
    for channel in channels:
        files[channel] = {int(p.stem.rsplit("s", 1)[1]): p
                          for p in (base / ("ch_" + channel)).glob(f"Z.DY_COSTH.{channel}.costh_cs.s*.dat")}
    seeds = sorted(set.intersection(*(set(v) for v in files.values())))
    if len(seeds) < 2:
        raise ValueError("need at least two complete independent runs")
    runs, hashes = [], {}
    for seed in seeds:
        total = None
        for channel in channels:
            path = files[channel][seed]
            a = np.loadtxt(path, comments="#")
            if not np.allclose(a[:, 0], np.linspace(-1, .95, 40)):
                raise ValueError("unexpected signed-angle binning")
            total = a[:, 3::2].copy() if total is None else total + a[:, 3::2]
            hashes[str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Fold the signed angle. Raw NNLOJET densities are fb per unit angle.
        runs.append((total[20:] + total[19::-1]) / 1000)
    with np.load(args.recoil_cache) as z:
        lo, hi, values = z["fo_ptw_lo"], z["fo_ptw_hi"], z["fo_ptw_A"]
        # These cached values are already pb/GeV. Use exact original bin edges.
        indices = [25, 35, 45, 55, 65, 80, 100]
        edges = np.r_[lo[indices[:-1]], hi[-1]]
        if edges[0] < 60:
            raise ValueError("public recoil comparison must be above the transition")
        recoil = np.stack([np.sum(values[:, a:b, :] * (hi[a:b] - lo[a:b])[None, :, None], axis=1)
                           / (edges[k + 1] - edges[k]) for k, (a, b) in enumerate(zip(indices[:-1], indices[1:]))], axis=1)
        recoil_seeds = z["fo_ptw_seeds"]
    record = dict(angle=dict(observable="absolute Collins-Soper angle", run="DY_COSTH",
                             source="NNLOJET v1.0.2 dy_costh_cs", channels=channels, seeds=seeds,
                             units="pb per unit angle", independent_of_moment_production=True,
                             fiducial="13 TeV; 66<mll<116 GeV; both lepton pT>27 GeV and |eta|<2.5; pre-QED leptons",
                             input_sha256=hashes),
                  recoil=dict(source="NNLOJET dy_profile_log30_hi, cached fo_ptw_A",
                              units="pb/GeV", independent_of_moment_production=False,
                              use="closure of the constrained hard spectrum, not independent physics validation",
                              cache_sha256=hashlib.sha256(Path(args.recoil_cache).read_bytes()).hexdigest()))
    np.savez_compressed(output, angle_edges=np.linspace(0, 1, 21), angle_per_run=np.asarray(runs),
                        angle_seeds=np.asarray(seeds), recoil_edges=edges, recoil_per_run=recoil,
                        recoil_seeds=recoil_seeds, reference_provenance_json=np.asarray(json.dumps(record, sort_keys=True)))
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {len(seeds)} independent angle runs and {len(recoil_seeds)} recoil runs to {output}")


if __name__ == "__main__":
    main()
