"""Compare moment and histogram refinement on a compatible full prior.

    python examples/zj_refinement.py prior.npz refinement_run
    python examples/plot_zj_refinement.py --results refinement_run --output refinement_plots

Use a compute host for full event samples. The plotter also redraws the released
aggregate results without access to the authors' event production.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import warnings

import numpy as np

for message in (
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
):
    warnings.filterwarnings("ignore", message=message, category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import zj_benchmark as bench
from maxent_upgrade.solver import _newton_maxent
from maxent_upgrade.provenance import file_hash


def merged_groups(counts, b, minimum=25):
    """Deterministic adjacent merging using prior counts only."""
    groups = {i * b + j: {i * b + j} for i in range(b) for j in range(i + 1)}
    xy = np.array([(i + 0.5, j + 0.5) for i in range(b) for j in range(b)])
    while len(groups) > 1:
        totals = {g: int(counts[list(c)].sum()) for g, c in groups.items()}
        g = min(groups, key=lambda k: (totals[k], k))
        if totals[g] >= minimum:
            break
        adjacent = []
        for h in groups:
            if h == g:
                continue
            distance = np.abs(xy[list(groups[g])][:, None] - xy[list(groups[h])]).sum(
                axis=2
            )
            if (distance == 1).any():
                adjacent.append(h)
        assert adjacent
        centre = xy[list(groups[g])].mean(axis=0)
        h = min(
            adjacent,
            key=lambda k: (
                float(np.sum((centre - xy[list(groups[k])].mean(axis=0)) ** 2)),
                k,
            ),
        )
        keep, discard = min(g, h), max(g, h)
        groups[keep] |= groups[discard]
        del groups[discard]
    members = [sorted(groups[g]) for g in sorted(groups)]
    mapping = np.full(b * b, -1, int)
    for k, cells in enumerate(members):
        mapping[cells] = k
    return mapping, members


def normalized_exp(score, weights):
    factor = np.exp(score - score.max())
    return factor / (weights @ factor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replicas", type=int, default=256)
    args = parser.parse_args()
    if args.replicas < 2:
        parser.error("at least two replicas are required")
    args.output.mkdir(exist_ok=False)
    data = HERE / "data/zj_benchmark"
    bench.DEFS = json.loads((data / "definitions.json").read_text())
    events = bench.load_prior(args.prior)
    p = events["weight"] / events["weight"].sum()
    H, names = bench.indicators(events)
    train = np.load(data / "training_fo.npz")
    central_target = bench.aggregate(train)
    x, y = events["ptj1"], events["ptj2"]
    cx = np.polynomial.chebyshev.chebvander(
        2 * np.log(x / 60) / np.log(1000 / 60) - 1, 4
    )
    cy = np.polynomial.chebyshev.chebvander(
        2 * np.log(y / 60) / np.log(1000 / 60) - 1, 4
    )
    models = []
    for j in range(5):
        indices = [(k, 0) for k in range(1, 5)] + [(0, k) for k in range(1, 5)]
        indices += [
            (k, degree_y) for k in range(1, j + 1) for degree_y in range(1, j + 1)
        ]
        models.append(
            dict(
                name="marg_D4" if j == 0 else f"D4_J{j}",
                kind="moments",
                indices=indices,
                Phi=np.column_stack(
                    [cx[:, k] * cy[:, degree_y] for k, degree_y in indices]
                ),
                constraints=len(indices),
            )
        )
    grid_records = []
    for b in (2, 3, 4, 5, 7, 10):
        edges = np.array(bench.DEFS["train_edges"][str(b)])
        ix = np.searchsorted(edges, x, side="right") - 1
        iy = np.searchsorted(edges, y, side="right") - 1
        cells = ix * b + iy
        counts = np.bincount(cells, minlength=b * b)
        mapping, members = merged_groups(counts, b)
        groups = mapping[cells]
        assert (groups >= 0).all()
        grouping = np.zeros((len(members), b * b))
        for k, v in enumerate(members):
            grouping[k, v] = 1
        raw = (
            central_target[f"joint_{b}"][:, :, 0].ravel() / central_target["norm"][0, 0]
        )
        assert abs(raw.sum() - 1) < 1e-10
        assert np.max(np.abs(raw[mapping < 0]), initial=0) < 1e-14
        models.append(
            dict(
                name=f"merged_B{b}",
                kind="histogram",
                bins=b,
                cells=cells,
                groups=groups,
                grouping=grouping,
                constraints=len(members) - 1,
            )
        )
        grid_records.append(
            dict(
                bins=b,
                counts=counts.reshape(b, b).tolist(),
                target_probabilities=raw.reshape(b, b).tolist(),
                unsupported_probability=float(raw[counts == 0].sum()),
                merged_cells=members,
                merged_counts=np.bincount(groups, minlength=len(members)).tolist(),
            )
        )

    def solve(model, prior, target, scale=0, warm=None):
        keep = prior > 0
        if model["kind"] == "moments":
            Phi = model["Phi"]
            mu = np.array(
                [
                    target["moments"][k, degree_y, scale] / target["norm"][0, scale]
                    for k, degree_y in model["indices"]
                ]
            )
            q, lam, ok = _newton_maxent(Phi[keep], prior[keep], mu, l2=1e-4, lam0=warm)
            if not ok:
                raise RuntimeError("moment fit failed")
            mean = prior[keep] @ Phi[keep]
            sd = np.maximum(
                np.sqrt(prior[keep] @ ((Phi[keep] - mean) ** 2) + 1e-30), 1e-12
            )
            correction = normalized_exp(Phi @ (lam / sd), prior)
            result = prior * correction
            assert np.max(np.abs(result[keep] - q)) < 1e-10
            residual = float(np.max(abs(result @ Phi - mu)))
            return result, correction, lam, residual
        g = model["groups"]
        target_cells = (
            target[f"joint_{model['bins']}"][:, :, scale].ravel()
            / target["norm"][0, scale]
        )
        target_groups = model["grouping"] @ target_cells
        prior_groups = np.bincount(g, weights=prior, minlength=len(target_groups))
        if (prior_groups <= 0).any():
            raise RuntimeError("merged group empty in replica")
        correction = (target_groups / prior_groups)[g]
        result = prior * correction
        result /= result.sum()
        residual = float(
            np.max(
                abs(
                    np.bincount(g, weights=result, minlength=len(target_groups))
                    - target_groups
                )
            )
        )
        assert residual < 1e-10
        return result, correction, None, residual

    start = time.monotonic()
    nm = len(models)
    predictions = np.empty((nm, 7, H.shape[1]))
    central_q = np.empty((nm, len(p)))
    corrections = np.empty_like(central_q)
    warm = []
    residuals = []
    for m, model in enumerate(models):
        for scale in range(7):
            q, c, lam, residual = solve(model, p, central_target, scale)
            predictions[m, scale] = q @ H
            if scale == 0:
                central_q[m] = q
                corrections[m] = c
                warm.append(lam)
                residuals.append(residual)
        print("Central", model["name"], model["constraints"], flush=True)
    # Archived arrays define the fixed diagnostic grid, not targets for a user prior.
    archive = np.load(data / "benchmark_results.npz")
    boots = np.full((nm, args.replicas, H.shape[1]), np.nan)
    corr_sum = np.zeros_like(corrections)
    corr_sq = np.zeros_like(corrections)
    common_counts = H[:, 15:79].sum(axis=0)
    common_cells = common_counts >= 50
    if not common_cells.any():
        raise ValueError(
            "no comparison cell contains 50 prior events; use a larger sample"
        )
    event_cell = np.argmax(H[:, 15:79], axis=1)
    assert np.all(H[:, 15:79].sum(axis=1) == 1)
    common_events = common_cells[event_cell]
    common_sum = np.zeros_like(corrections)
    common_sq = np.zeros_like(corrections)
    failures = []
    success = np.zeros(nm, int)
    rng = np.random.default_rng(15112026)
    for rep in range(args.replicas):
        ids = rng.integers(0, len(train["seeds"]), len(train["seeds"]))
        target = bench.aggregate(train, ids)
        mult = rng.poisson(1, len(p))
        pr = p * mult
        pr /= pr.sum()
        for m, model in enumerate(models):
            try:
                q, c, _, _ = solve(model, pr, target, warm=warm[m])
                boots[m, rep] = q @ H
                # Shape of the learned correction on fixed original event locations.
                c = c / (p @ c)
                ratio = c / corrections[m]
                corr_sum[m] += ratio
                corr_sq[m] += ratio**2
                success[m] += 1
                # Additional shape diagnostic on the original, fixed 19-cell region.
                ratio_common = (
                    ratio
                    * central_q[m, common_events].sum()
                    / (p[common_events] @ c[common_events])
                )
                common_sum[m] += ratio_common
                common_sq[m] += ratio_common**2
            except (RuntimeError, ValueError) as exc:
                failures.append(dict(replica=rep, method=model["name"], error=str(exc)))
        if (rep + 1) % 32 == 0:
            print("Replicas", rep + 1, "elapsed", time.monotonic() - start, flush=True)
    variance = (corr_sq - corr_sum**2 / success[:, None]) / (success[:, None] - 1)
    correction_spread = np.sqrt(np.maximum(variance, 0))
    common_variance = (common_sq - common_sum**2 / success[:, None]) / (
        success[:, None] - 1
    )
    common_spread = np.sqrt(np.maximum(common_variance, 0))
    # Outcome evaluation starts after all new fits and resamples are fixed.
    valid = np.load(data / "validation_fo.npz")
    vr = bench.fo_vector(valid)
    vn = valid["norm"][:, 0, :]
    reference = bench.transformed(vr.mean(0) / vn.mean(0)[:, None])
    predictions = bench.transformed(predictions)
    boots = bench.transformed(boots)
    vrng = np.random.default_rng(15112027)
    vrep = []
    for _ in range(args.replicas):
        ids = vrng.integers(0, len(vn), len(vn))
        vrep.append(bench.transformed(vr[ids].mean(0) / vn[ids].mean(0)[:, None])[0])
    vrep = np.asarray(vrep)
    mask = common_cells
    results = []
    for m, model in enumerate(models):
        good = np.isfinite(boots[m]).all(axis=1)
        distances = 0.5 * np.abs(
            boots[m, good, 15:79][:, mask] - vrep[good, 15:79][:, mask]
        ).sum(1)
        relative_spread = correction_spread[m]
        order = np.argsort(relative_spread)
        cdf = np.cumsum(central_q[m, order])
        quantiles = np.interp([0.5, 0.9, 0.99], cdf, relative_spread[order])
        results.append(
            dict(
                name=model["name"],
                constraints=model["constraints"],
                successes=int(good.sum()),
                tv2d_supported=float(
                    0.5
                    * abs(
                        predictions[m, 0, 15:79][mask] - reference[0, 15:79][mask]
                    ).sum()
                ),
                tv2d_all=float(
                    0.5 * abs(predictions[m, 0, 15:79] - reference[0, 15:79]).sum()
                ),
                tv2d_spread=float(distances.std(ddof=1)),
                correction_relative_rms=float(
                    np.sqrt(central_q[m] @ relative_spread**2)
                ),
                correction_relative_quantiles=quantiles.tolist(),
                correction_relative_rms_common=float(
                    np.sqrt(
                        (
                            central_q[m, common_events]
                            @ common_spread[m, common_events] ** 2
                        )
                        / central_q[m, common_events].sum()
                    )
                ),
                cut_residuals=(predictions[m, 0, :3] / reference[0, :3] - 1).tolist(),
                cut_fit_errors=boots[m, good, :3].std(0, ddof=1).tolist(),
                neff=float(1 / (central_q[m] @ central_q[m])),
                closure_abs=residuals[m],
            )
        )
    # For a fair uncertainty comparison, retain all failures explicitly.
    summary = dict(
        prior_events=len(p),
        replicas=args.replicas,
        common_cells=int(mask.sum()),
        common_reference_probability=float(reference[0, 15:79][mask].sum()),
        models=results,
        grids=grid_records,
        failures=failures,
        seconds=time.monotonic() - start,
        protocol_sha256=hashlib.sha256(
            (HERE / "data/zj_refinement/PROTOCOL.txt").read_bytes()
        ).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        solver_sha256=file_hash(Path(_newton_maxent.__code__.co_filename)),
        input_files={
            path.name: file_hash(path)
            for path in [
                args.prior,
                data / "training_fo.npz",
                data / "validation_fo.npz",
                data / "definitions.json",
            ]
        },
        interpretation="Correction-factor spread is distinct from uncertainty of refitted-sample predictions; no monotonic trend is imposed.",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.output / "results.npz",
        methods=np.array([m["name"] for m in models]),
        predictions=predictions,
        bootstrap=boots,
        reference=reference,
        reference_replicas=vrep,
        reference_covariance=archive["reference_covariance"],
        common_mask=mask,
        correction_relative_spread=correction_spread,
        correction_relative_spread_common=common_spread,
        common_events=common_events,
        central_weights=central_q,
        prior_weights=p,
        grid_counts=common_counts.reshape(8, 8),
    )
    for row in results:
        print(
            f"{row['name']:12s} K={row['constraints']:2d}  "
            f"distance={100 * row['tv2d_supported']:.3f}%  "
            f"correction spread={100 * row['correction_relative_rms_common']:.3f}%  "
            f"replicas={row['successes']}/{args.replicas}",
            flush=True,
        )


if __name__ == "__main__":
    main()
