"""Render all frozen benchmark results; input is aggregate caches only."""
from pathlib import Path
import json
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def json_data(value):
    """Missing fit diagnostics use JSON null; numeric caches retain NaN."""
    if isinstance(value, dict):
        return {key: json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_data(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "data/zj_benchmark"))
    parser.add_argument("--output", default="zj_benchmark_plots")
    parser.add_argument("--results", help="directory of a new fit; defaults to the released caches")
    args = parser.parse_args(argv)
    ROOT = Path(args.data)
    RESULTS = Path(args.results) if args.results else ROOT
    OUTPUT = Path(args.output)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    r = np.load(RESULTS / 'benchmark_results.npz')
    summary = json.loads((RESULTS / 'benchmark_summary.json').read_text())
    defs = json.loads((ROOT / 'definitions.json').read_text())
    methods = r['methods'].tolist()
    names = r['names'].tolist()
    pred = r['predictions'][:, 0]
    boot = r['bootstrap']
    ref = r['reference'][0]
    referr = np.sqrt(np.diag(r['reference_covariance']))
    mask = r['local_counts'].ravel() >= 50
    # A separate resample of the validation runs gives the numerical spread of
    # distances, paired with each already-frozen fit replica. It does not select fits.
    v = np.load(ROOT / 'validation_fo.npz')
    vv = np.stack([v['valid_xy'][:, i, j, 0] for i in range(8) for j in range(8)], axis=1)
    vn = v['norm'][:, 0, 0]
    vrng = np.random.default_rng(39114)
    vrep = []
    for _ in range(boot.shape[1]):
        ids = vrng.integers(0, len(vn), len(vn))
        vrep.append(vv[ids].mean(axis=0) / vn[ids].mean())
    vrep = np.array(vrep)
    rows = []
    details = {}
    for m, name in enumerate(methods):
        record = summary['methods'][m]
        good = np.isfinite(boot[m]).all(axis=1)
        err = np.std(boot[m, good], axis=0, ddof=1) if good.sum() > 1 else np.full(len(ref), np.nan)
        row = dict(method=name, constraints=record['constraints'], bootstrap_successes=int(good.sum()),
                   neff=record.get('neff', np.nan), neff_retained=record.get('neff_retained', np.nan),
                   max_weight_factor=record.get('max_weight_factor', np.nan),
                   fit_and_7_scales_seconds=record['central_and_scales_seconds'],
                   tv2d_all=.5 * float(abs(pred[m, 15:79] - ref[15:79]).sum()),
                   tv2d_supported=.5 * float(abs(pred[m, 15:79][mask] - ref[15:79][mask]).sum()),
                   tv2d_numerical_spread=float(np.std(.5 * abs(boot[m, good, 15:79][:, mask] - vrep[good][:, mask]).sum(axis=1), ddof=1)) if good.sum() > 1 else np.nan,
                   balance_tv_low=.5 * float(abs(pred[m, 3:9] - ref[3:9]).sum()),
                   balance_tv_high=.5 * float(abs(pred[m, 9:15] - ref[9:15]).sum()),
                   angular_tv=.5 * float(abs(pred[m, 79:86] - ref[79:86]).sum()))
        for k in range(3):
            row.update({f'cut{k}_efficiency': pred[m, k], f'cut{k}_fit_stat': err[k],
                        f'cut{k}_relative_error': pred[m, k] / ref[k] - 1,
                        f'cut{k}_numerical_pull': (pred[m, k] - ref[k]) / np.hypot(err[k], referr[k])})
        rows.append(row)
        details[name] = dict(bootstrap_successes=int(good.sum()), stat_errors=err.tolist(),
                             deterministic_scale_min=r['predictions'][m].min(axis=0).tolist(),
                             deterministic_scale_max=r['predictions'][m].max(axis=0).tolist(),
                             matched_scale_cut_residuals=(r['predictions'][m, :, :3] / r['reference'][:, :3] - 1).tolist(),
                             matched_scale_tv2d_supported=(.5 * abs(r['predictions'][m, :, 15:79][:, mask] - r['reference'][:, 15:79][:, mask]).sum(axis=1)).tolist(),
                             central_errors=record['errors'])
    with (OUTPUT / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    support = []
    ve = defs['validation_edges']
    for i in range(8):
        for j in range(8):
            cell = i * 8 + j
            entry = dict(ptj1_lo=ve[i], ptj1_hi=ve[i + 1], ptj2_lo=ve[j], ptj2_hi=ve[j + 1],
                         prior_events=int(r['local_counts'][i, j]), included_in_tv=bool(mask[cell]),
                         reference_probability=ref[15 + cell], reference_stat=referr[15 + cell])
            for m, name in enumerate(methods):
                entry[f'{name}_probability'] = r['local_mass'][m, i, j]
                entry[f'{name}_neff'] = r['local_neff'][m, i, j]
                entry[f'{name}_max_weight_factor'] = r['local_max_weight_factor'][m, i, j]
            support.append(entry)
    with (OUTPUT / 'local_support.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(support[0]))
        writer.writeheader()
        writer.writerows(support)

    # Paired increments quantify truncation sensitivity, without changing any model.
    sequence = ['marg_D4', 'D4_J1', 'D4_J2', 'D4_J3', 'D4_J4']
    increments = []
    for j, (old, new) in enumerate(zip(sequence[:-1], sequence[1:]), 1):
        a, b = methods.index(old), methods.index(new)
        delta = boot[b] - boot[a]
        good = np.isfinite(delta).all(axis=1)
        se = delta[good].std(axis=0, ddof=1)
        block = np.array([(m == j or n == j) and m <= j and n <= j
                          for m in range(1, 5) for n in range(1, 5)])
        gap = r['joint_residuals'][a, 0, block]
        gapboot = r['bootstrap_joint_residuals'][a][:, block]
        gcov = np.atleast_2d(np.cov(gapboot, rowvar=False, ddof=1))
        increments.append(dict(previous=old, next=new,
                               added_moment_gap=gap.tolist(), added_moment_stat=np.sqrt(np.diag(gcov)).tolist(),
                               added_moment_covariance=gcov.tolist(),
                               cut_change=(pred[b, :3] - pred[a, :3]).tolist(), cut_change_stat=se[:3].tolist(),
                               cut_change_relative_to_reference=((pred[b, :3] - pred[a, :3]) / ref[:3]).tolist(),
                               balance_change_tv=[.5 * float(abs(pred[b, sl] - pred[a, sl]).sum()) for sl in (slice(3, 9), slice(9, 15))]))

    analysis = dict(reference=dict(cuts=ref[:3].tolist(), cut_stat=referr[:3].tolist(),
                                   balance_low=ref[3:9].tolist(), balance_high=ref[9:15].tolist(),
                                   balance_low_stat=referr[3:9].tolist(), balance_high_stat=referr[9:15].tolist()),
                    common_mask=dict(cells=int(mask.sum()), prior_events_minimum=50,
                                     reference_mass=float(ref[15:79][mask].sum()),
                                     excluded_reference_mass=float(ref[15:79][~mask].sum())),
                    methods=details, joint_increments=increments)
    analysis['angular_reference_status'] = summary.get('angular_reference_status')
    analysis['primary_jet_quantities_status'] = summary.get('primary_jet_quantities_status')
    (OUTPUT / 'analysis.json').write_text(json.dumps(json_data(analysis), indent=2, allow_nan=False) + '\n')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    groups = [('Marginal moments', ['marg_D4'], '#b56b29', 's'),
              ('Joint moments, fixed marginals', [f'D4_J{j}' for j in range(1, 5)], '#2466a0', 'o'),
              ('Joint moments, full degree', [f'D{j}_J{j}' for j in range(1, 5)], '#3891a6', '^'),
              ('Marginal histogram IPF', [f'ipf_B{b}' for b in (2, 3, 4, 5, 7, 10)], '#be7a1a', 'v'),
              ('Joint histogram ratio', [f'ratio_B{b}' for b in (2, 3, 4, 5, 7, 10)], '#754e9a', 'D')]
    for label, group, color, marker in groups:
        rr = [rows[methods.index(n)] for n in group]
        ax[0, 0].errorbar([x['constraints'] for x in rr], [100 * x['tv2d_supported'] for x in rr],
                          yerr=[100 * x['tv2d_numerical_spread'] for x in rr],
                          color=color, marker=marker, label=label, lw=1, ms=4, capsize=2)
    ax[0, 0].axhline(100 * rows[0]['tv2d_supported'], color='0.5', ls=':', label='Prior')
    ax[0, 0].set(xlabel='Independent shape constraints', ylabel='Held-out 2D TV on populated cells [%]', xlim=(0, 56))
    ax[0, 0].legend(fontsize=8)
    selected = ['prior', 'marg_D4', 'D4_J2', 'D4_J4', 'ipf_B5', 'ratio_B5']
    colors = ['0.5', '#b56b29', '#3891a6', '#2466a0', '#be7a1a', '#754e9a']
    labels = ['Prior', 'Marginal moments D=4', 'Joint moments J=2', 'Joint moments J=4', 'Marginal IPF B=5', 'Joint ratio B=5']
    for h, (name, color, label) in enumerate(zip(selected, colors, labels)):
        m = methods.index(name)
        err = np.asarray(details[name]['stat_errors'])
        xs = np.arange(3) + (h - 2.5) * .105
        ax[0, 1].errorbar(xs, 100 * (pred[m, :3] / ref[:3] - 1),
                          yerr=100 * np.hypot(err[:3], referr[:3]) / ref[:3],
                          marker='o', ls='none', color=color, ms=4, label=label, capsize=2)
    ax[0, 1].axhline(0, color='0.3', lw=.7)
    ax[0, 1].set_xticks(range(3), ['(125, 80)', '(180, 110)', '(300, 150)'])
    ax[0, 1].set(xlabel=r'Withheld cuts $(p_{T,j_1},p_{T,j_2})$ [GeV]', ylabel='Efficiency residual [%]')
    ax[0, 1].legend(fontsize=7.5, ncol=2)
    edges = np.asarray(defs['balance_edges'])
    mid = (edges[:-1] + edges[1:]) / 2
    for k in range(2):
        a = ax[1, k]
        sl = slice(3 + 6 * k, 9 + 6 * k)
        a.errorbar(mid, ref[sl], yerr=referr[sl], color='black', fmt='o', ms=3, label='Independent fixed order')
        for name, color, label in zip(selected, colors, labels):
            m = methods.index(name)
            a.stairs(pred[m, sl], edges, color=color, lw=1.2, label=label)
        lo, hi = defs['balance_intervals'][k]
        a.set(xlabel=r'$p_{T,j_2}/p_{T,j_1}$', ylabel='Conditional bin probability',
              title=rf'${lo}<p_{{T,j_1}}<{hi}$ GeV', xlim=(0, 1))
    ax[1, 0].legend(fontsize=7.5)
    fig.suptitle('Controlled LO Z+2-jet shape comparison: identical prior and 16M training points', fontsize=12)
    failed = [x['name'] for x in summary['methods'] if '0' in x['errors']]
    footer = ('Central fits with absent prior support: ' + ', '.join(failed) + '. ' if failed else '')
    footer += 'Error bars show numerical fitting and independent-reference errors; scales are saved separately.'
    fig.text(.5, .005, footer, ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, .98))
    fig.savefig(OUTPUT / 'benchmark_overview.pdf')
    fig.savefig(OUTPUT / 'benchmark_overview.png', dpi=180)
    if analysis.get('angular_reference_status'):
        print('Recorded angular-reference status:', analysis['angular_reference_status'])
    print('Reference cut efficiencies:', ref[:3], '+/-', referr[:3])
    for row in rows:
        print(f"{row['method']:14s} k={row['constraints']:2d} TV={row['tv2d_supported']:.5f} "
              f"cuts={[round(float(100 * row[f'cut{k}_relative_error']), 2) for k in range(3)]} "
              f"neff={row['neff']:.1f} boot={row['bootstrap_successes']}/{boot.shape[1]}")
    print('Common populated cells:', analysis['common_mask'])


if __name__ == "__main__":
    main()
