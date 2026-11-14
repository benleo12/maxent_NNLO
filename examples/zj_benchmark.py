"""Refit the frozen Z+2-jet comparison on a compatible prior.

    python examples/zj_benchmark.py my_zj_prior.npz benchmark_run

Uses the committed training and independent validation inputs. Methods and
resolution grids are fixed before reading validation. Full event work belongs
on a compute host; the companion plot script reads only compact caches.
"""
from pathlib import Path
import json
import time
import hashlib
import sys
import warnings
import argparse
import platform
import numpy as np

# Affected macOS Accelerate builds can raise stale flags after finite GEMMs.
# All explicit finite-value and convergence checks remain active.
if platform.system() == "Darwin":
    for message in ('divide by zero encountered in matmul', 'overflow encountered in matmul',
                    'invalid value encountered in matmul'):
        warnings.filterwarnings('ignore', message=message, category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import maxent_upgrade
from maxent_upgrade import upgrade
from maxent_upgrade.provenance import content_hash, file_hash

DEFS = {}
SPECS = [('prior', 'prior', 0, 0), ('marg_D4', 'maxent', 4, 0)]
SPECS += [(f'D4_J{j}', 'maxent', 4, j) for j in range(1, 5)]
SPECS += [(f'D{j}_J{j}', 'maxent', j, j) for j in range(1, 4)]
SPECS += [(f'{kind}_B{b}', kind, b, 0) for kind in ('ipf', 'ratio') for b in (2, 3, 4, 5, 7, 10)]


def load_prior(path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in ("weight", "ptj1", "ptj2", "njet", "pimdphi")}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or any(value.ndim != 1 for value in arrays.values()):
        raise ValueError("benchmark input arrays must be one-dimensional and have matching lengths")
    w = arrays["weight"].astype(float)
    x, y = arrays["ptj1"].astype(float), arrays["ptj2"].astype(float)
    selected = (arrays["njet"] >= 2) & (y >= 60) & (x < 1000)
    if not selected.any():
        raise ValueError("no prior event lies in the common hard two-jet region")
    if not np.isfinite(w[selected]).all() or (w[selected] <= 0).any():
        raise ValueError("the selected prior must have finite strictly positive weights")
    events = dict(ptj1=x[selected], ptj2=y[selected], weight=w[selected],
                  aco=arrays["pimdphi"][selected].astype(float))
    if not all(np.isfinite(value).all() for value in events.values()):
        raise ValueError("selected benchmark observables must be finite")
    if not np.all(events["ptj1"] >= events["ptj2"]):
        raise ValueError("ptj1 and ptj2 must be the ordered leading and second jet momenta")
    if not np.all((events["aco"] >= 0) & (events["aco"] <= np.pi)):
        raise ValueError("pimdphi must be pi minus the dilepton azimuthal separation in radians")
    return events


def aggregate(data, indices=None):
    if indices is None:
        indices = np.arange(len(data['seeds']))
    return {k: np.asarray(data[k])[indices].mean(axis=0) for k in data.files
            if k not in ('seeds', 'points_per_run', 'minutes')}


def fit(spec, events, target, scale=0, lam0=None):
    name, kind, degree, joint = spec
    w = events['weight']
    p = w / w.sum()
    if kind == 'prior':
        return p, dict(constraints=0, closure_abs=0, iterations=0)
    norm = target['norm'][0, scale]
    if kind == 'maxent':
        mu = target['moments'][:, :, scale] / norm
        maps = {o: dict(range=(60., 1000.), map='log') for o in ('ptj1', 'ptj2')}
        config = dict(born=maps, fixed_orders=dict(ptj1=degree, ptj2=degree),
                      moment_selection=False, L2=1e-4, keep_features=True)
        moments = dict(born=dict(ptj1=dict(values=mu[1:, 0], errors=np.zeros(4)),
                                 ptj2=dict(values=mu[0, 1:], errors=np.zeros(4))))
        if joint:
            config['mixed'] = dict(jets=dict(observables=['ptj1', 'ptj2'],
                                             range={o: (60., 1000.) for o in maps}, map={o: 'log' for o in maps}, n=joint))
            moments['mixed'] = dict(jets=dict(values={(m, n): float(mu[m, n])
                                                      for m in range(1, joint + 1) for n in range(1, joint + 1)}, errors={}, rate=1.))
        if lam0 is not None:
            config['lam0'] = lam0
        result = upgrade(events, moments, config)
        q = result.weights
        assert np.isfinite(q).all() and (q > 0).all() and abs(q.sum() - 1) < 1e-10
        report = result.report
        # Recompute the absolute residual without dividing by near-zero moments.
        cx = np.polynomial.chebyshev.chebvander(2 * np.log(events['ptj1'] / 60) / np.log(1000 / 60) - 1, 4)
        cy = np.polynomial.chebyshev.chebvander(2 * np.log(events['ptj2'] / 60) / np.log(1000 / 60) - 1, 4)
        residual = [q @ cx[:, n] - mu[n, 0] for n in range(1, degree + 1)]
        residual += [q @ cy[:, n] - mu[0, n] for n in range(1, degree + 1)]
        residual += [q @ (cx[:, m] * cy[:, n]) - mu[m, n] for m in range(1, joint + 1) for n in range(1, joint + 1)]
        phi = report['Phi'][:, 1:]
        sd = np.sqrt(p @ ((phi - p @ phi)**2))
        z = (phi - q @ phi) / sd
        eig = np.linalg.eigvalsh((q[:, None] * z).T @ z)
        return q, dict(constraints=2 * degree + joint**2, closure_abs=float(np.max(np.abs(residual))),
                       lam=np.asarray(report['lam']).tolist(),
                       feature_condition=float(eig[-1] / max(eig[0], 1e-30)),
                       regularized_condition=float((eig[-1] + 1e-4) / (eig[0] + 1e-4)))
    b = degree
    edges = np.asarray(DEFS['train_edges'][str(b)])
    ix = np.searchsorted(edges, events['ptj1'], side='right') - 1
    iy = np.searchsorted(edges, events['ptj2'], side='right') - 1
    assert np.all((ix >= 0) & (ix < b) & (iy >= 0) & (iy < b))
    if kind == 'ratio':
        cell = ix * b + iy
        target_cell = target[f'joint_{b}'][:, :, scale].ravel() / norm
        prior_cell = np.bincount(cell, weights=p, minlength=b * b)
        missing = (prior_cell == 0) & (target_cell > 1e-14)
        if missing.any():
            raise RuntimeError(f'{missing.sum()} positive-target cells without prior support; '
                               f'target mass={target_cell[missing].sum():.9g}; '
                               f'cells={np.flatnonzero(missing).tolist()}')
        ratio = np.divide(target_cell, prior_cell, out=np.zeros_like(target_cell), where=prior_cell > 0)
        q = p * ratio[cell]
        q /= q.sum()
        residual = np.bincount(cell, weights=q, minlength=b * b) - target_cell
        return q, dict(constraints=int((target_cell > 0).sum() - 1), closure_abs=float(abs(residual).max()), iterations=1)
    tx = target[f'train_x_{b}'][:, scale] / norm
    ty = target[f'train_y_{b}'][:, scale] / norm
    q = p.copy()
    for iteration in range(10000):
        for index, targ in ((ix, tx), (iy, ty)):
            current = np.bincount(index, weights=q, minlength=b)
            missing = (current == 0) & (targ > 1e-14)
            if missing.any():
                raise RuntimeError('Positive marginal target without prior support; '
                                   f'target mass={targ[missing].sum():.9g}; '
                                   f'bins={np.flatnonzero(missing).tolist()}')
            factor = np.divide(targ, current, out=np.zeros_like(targ), where=current > 0)
            q *= factor[index]
        residual = max(abs(np.bincount(ix, weights=q, minlength=b) - tx).max(),
                       abs(np.bincount(iy, weights=q, minlength=b) - ty).max())
        if residual < 1e-10:
            break
    else:
        raise RuntimeError(f'IPF failed after 10000 iterations, residual={residual}')
    return q / q.sum(), dict(constraints=2 * (b - 1), closure_abs=float(residual), iterations=iteration + 1)


def indicators(events):
    x, y = events['ptj1'], events['ptj2']
    columns, names = [], []

    def add(name, col):
        names.append(name)
        columns.append(col)
    for i, (a, b) in enumerate(DEFS['cuts']):
        add(f'cut_{i}', (x > a) & (y > b))
    for k, (a, b) in enumerate(DEFS['balance_intervals']):
        inside = (x >= a) & (x < b)
        edges = DEFS['balance_edges']
        ratio = y / x
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            add(f'balance_{k}_{i}', inside & (ratio >= lo) & (ratio < hi))
    e = DEFS['validation_edges']
    for i, (lo, hi) in enumerate(zip(e[:-1], e[1:])):
        for j, (a, b) in enumerate(zip(e[:-1], e[1:])):
            add(f'xy_{i}_{j}', (x >= lo) & (x < hi) & (y >= a) & (y < b))
    a = events['aco']
    for i, (lo, hi) in enumerate(zip(DEFS['aco_edges'][:-1], DEFS['aco_edges'][1:])):
        add(f'aco_{i}', (a >= lo) & ((a < hi) if i < 6 else (a <= hi)))
    return np.array(columns, dtype=float).T, names


def fo_vector(data):
    # shape run, scale, observable, with exactly the same observable order.
    a = [data[f'cut_{i}'][:, 0, :] for i in range(3)]
    a += [data[f'balance_{j}'][:, i, :] for j in range(2) for i in range(6)]
    a += [data['valid_xy'][:, i, j, :] for i in range(8) for j in range(8)]
    a += [data['aco'][:, i, :] for i in range(7)]
    return np.stack(a, axis=-1)


def transformed(a):
    """Conditional balance distributions, then their two interval rates."""
    a = np.array(a, copy=True)
    rates = []
    for j in range(2):
        sl = slice(3 + 6 * j, 9 + 6 * j)
        total = a[..., sl].sum(axis=-1)
        rates.append(total)
        a[..., sl] /= total[..., None]
    return np.concatenate([a, np.stack(rates, axis=-1)], axis=-1)


def main(argv=None):
    global DEFS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prior", help="selected-compatible arrays: weight, ptj1, ptj2, njet, pimdphi")
    parser.add_argument("output", help="new directory for fitted predictions and aggregate diagnostics")
    parser.add_argument("--data", default=str(ROOT / "examples/data/zj_benchmark"))
    parser.add_argument("--bootstrap", type=int, default=32)
    args = parser.parse_args(argv)
    if args.bootstrap < 2:
        parser.error("at least two replicas are required")
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    DEFS = json.loads((data / "definitions.json").read_text())
    start = time.monotonic()
    events = load_prior(args.prior)
    p = events['weight'] / events['weight'].sum()
    H, names = indicators(events)
    cx = np.polynomial.chebyshev.chebvander(2 * np.log(events['ptj1'] / 60) / np.log(1000 / 60) - 1, 4)
    cy = np.polynomial.chebyshev.chebvander(2 * np.log(events['ptj2'] / 60) / np.log(1000 / 60) - 1, 4)
    joint_features = np.column_stack([cx[:, m] * cy[:, n] for m in range(1, 5) for n in range(1, 5)])
    train = np.load(data / 'training_fo.npz')
    target = aggregate(train)
    predictions = np.full((len(SPECS), 7, H.shape[1]), np.nan)
    joint_residuals = np.full((len(SPECS), 7, 16), np.nan)
    weights = np.full((len(SPECS), len(p)), np.nan)
    records = []
    for m, spec in enumerate(SPECS):
        t = time.monotonic()
        count = (0 if spec[1] == 'prior' else 2 * spec[2] + spec[3]**2 if spec[1] == 'maxent'
                 else 2 * (spec[2] - 1) if spec[1] == 'ipf' else spec[2] * (spec[2] + 1) // 2 - 1)
        record = dict(name=spec[0], kind=spec[1], constraints=count, errors={})
        for s in range(7):
            try:
                q, meta = fit(spec, events, target, s)
                predictions[m, s] = q @ H
                joint_residuals[m, s] = q @ joint_features - target['moments'][1:, 1:, s].ravel() / target['norm'][0, s]
                if s == 0:
                    weights[m] = q
                    record.update(meta)
                    record.update(neff=float(1 / (q @ q)), neff_retained=float((p @ p) / (q @ q)),
                                  max_weight_factor=float((q / p).max()))
            except (RuntimeError, ValueError) as exc:
                record['errors'][str(s)] = str(exc)
        record['central_and_scales_seconds'] = time.monotonic() - t
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != 'lam'}), flush=True)
    np.savez_compressed(output / 'fit_checkpoint.npz', predictions=predictions, weights=weights)
    (output / 'fit_checkpoint.json').write_text(json.dumps(records, indent=2) + '\n')

    rng = np.random.default_rng(73113)
    boots = np.full((len(SPECS), args.bootstrap, H.shape[1]), np.nan)
    boot_joint_residuals = np.full((len(SPECS), args.bootstrap, 16), np.nan)
    boot_neff = np.full((len(SPECS), 32), np.nan)
    bootstrap_errors = []
    for r in range(args.bootstrap):
        indices = rng.integers(0, len(train['seeds']), len(train['seeds']))
        target_rep = aggregate(train, indices)
        mult = rng.poisson(1, len(p))
        keep = mult > 0
        ev = {k: v[keep].copy() for k, v in events.items()}
        ev['weight'] *= mult[keep]
        for m, spec in enumerate(SPECS):
            try:
                q, meta = fit(spec, ev, target_rep, lam0=records[m].get('lam'))
                boots[m, r] = q @ H[keep]
                boot_joint_residuals[m, r] = q @ joint_features[keep] - target_rep['moments'][1:, 1:, 0].ravel() / target_rep['norm'][0, 0]
                # Count duplicated original events as one statistical unit here.
                boot_neff[m, r] = 1 / (q @ q)
            except (RuntimeError, ValueError) as exc:
                bootstrap_errors.append(dict(replica=r, method=spec[0], error=str(exc)))
        print(f'Bootstrap {r + 1}/{args.bootstrap} complete', flush=True)
    # Validation is first opened after every fit and bootstrap model is frozen.
    valid = np.load(data / 'validation_fo.npz')
    vraw = fo_vector(valid)
    vnorm = valid['norm'][:, 0, :]
    reference = vraw.mean(axis=0) / vnorm.mean(axis=0)[:, None]
    # Delete-one jackknife retains common numerator/denominator correlations.
    n = len(vnorm)
    jack = (vraw.sum(axis=0)[None, :, :] - vraw) / (vnorm.sum(axis=0)[None, :] - vnorm)[:, :, None]
    vfinal = transformed(reference)
    jackfinal = transformed(jack)
    centered = jackfinal[:, 0] - jackfinal[:, 0].mean(axis=0)
    covariance = (n - 1) / n * centered.T @ centered
    pred_final = transformed(predictions)
    boot_final = transformed(boots)
    # Local support and effective counts on the held-out 8x8 grid.
    grid = H[:, 15:79]
    counts = grid.sum(axis=0)
    mass = weights @ grid
    sumsq = (weights**2) @ grid
    local_neff = np.divide(mass**2, sumsq, out=np.zeros_like(mass), where=sumsq > 0)
    max_factor = np.array([[np.max((q / p)[grid[:, j] > 0]) if counts[j] else np.nan
                          for j in range(64)] for q in weights])
    np.savez_compressed(output / 'benchmark_results.npz',
                        methods=np.array([x[0] for x in SPECS]), names=np.array(names + ['interval_rate_0', 'interval_rate_1']),
                        predictions=pred_final, raw_predictions=predictions, bootstrap=boot_final,
                        joint_residuals=joint_residuals, bootstrap_joint_residuals=boot_joint_residuals,
                        reference=vfinal, reference_covariance=covariance, reference_jackknife=jackfinal,
                        local_counts=counts.reshape(8, 8), local_neff=local_neff.reshape(-1, 8, 8),
                        local_max_weight_factor=max_factor.reshape(-1, 8, 8), local_mass=mass.reshape(-1, 8, 8),
                        bootstrap_neff=boot_neff, prior_neff=1 / (p @ p), prior_events=len(p))
    summary = dict(protocol_sha256=hashlib.sha256((data / 'PROTOCOL.txt').read_bytes()).hexdigest(),
                   original_study_package_commit='b437a68', prior=Path(args.prior).name, prior_events=len(p),
                   implementation_sha256=content_hash({p.name: file_hash(p) for p in Path(maxent_upgrade.__file__).parent.glob('*.py')}),
                   input_files=[dict(name=p.name, sha256=file_hash(p)) for p in
                                [Path(args.prior), data / 'training_fo.npz', data / 'validation_fo.npz', data / 'definitions.json']],
                   prior_neff=float(1 / (p @ p)), methods=records, bootstrap_errors=bootstrap_errors,
                   total_seconds=time.monotonic() - start, statistical_replicas=args.bootstrap,
                   sparse_mask_min_prior_events=50)
    (output / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('COMPLETE ' + str(summary['total_seconds']) + ' seconds', flush=True)


if __name__ == '__main__':
    main()
