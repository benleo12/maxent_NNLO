"""Redraw the independent W prediction example from compact public caches only."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

EDGES = np.array([0, .21, .42, .63, .84, 1.05, 1.37, 1.52, 1.74, 1.95, 2.18, 2.5])
COLORS = {'P': '#777777', 'B': '#28649b', 'A': '#d48a16', 'selected': '#b52230', 'J2': '#8f3c8f', 'J4': '#b52230'}
STYLES = {'P': ':', 'B': '--', 'A': '-.', 'selected': '-', 'J2': '--', 'J4': '-'}


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
                         'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.linewidth': .65, 'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.top': True, 'ytick.right': True, 'mathtext.fontset': 'stix', 'savefig.bbox': 'tight'})


def envelope(ax, edges, scales, color, alpha=.15, offset=None):
    x = np.asarray(scales)
    if offset is not None:
        x = x - np.asarray(offset)[None, :]
    lo = x.min(0)
    hi = x.max(0)
    ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step='post', color=color, alpha=alpha, linewidth=0)


def main_figure(data, out):
    asym = json.loads((data / 'charge_asymmetry.json').read_text())
    selected = {ch: json.loads((data / ('w_%s_validation.json' % ch)).read_text())['selected'] for ch in ['wp', 'wm']}
    names = ['P', 'B', 'A', 'selected'] if any(x != 'A' for x in selected.values()) else ['P', 'B', 'selected']
    fig, axs = plt.subplots(2, 2, figsize=(6.35, 3.5), sharex='col', gridspec_kw={'height_ratios': [2.05, 1.]})
    mids = .5 * (EDGES[1:] + EDGES[:-1])
    for col, cut in enumerate([20, 30]):
        tag = 'eta%d' % cut
        top, bottom = axs[:, col]
        ref = np.array(asym['reference'][tag]['coherent_scales'])
        rs = np.array(asym['reference'][tag]['stat'])
        for ax in [top, bottom]:
            ax.set_xlim(0, 2.5)
            ax.tick_params(which='both', length=3)
        envelope(top, EDGES, ref, '#777777', .22)
        envelope(bottom, EDGES, ref, '#777777', .22, ref[0])
        top.errorbar(mids, ref[0], yerr=rs, fmt='o', ms=2.2, lw=.65, color='black', zorder=5)
        bottom.errorbar(mids, np.zeros(len(mids)), yerr=rs, fmt='o', ms=2.2, lw=.65, color='black', zorder=5)
        for name in names:
            ys = np.array(asym[name][tag]['coherent_scales'])
            stat = np.array(asym[name][tag]['stat'])
            if name == 'selected':
                envelope(top, EDGES, ys, COLORS[name], .16)
                envelope(bottom, EDGES, ys, COLORS[name], .16, ref[0])
                top.errorbar(mids + .012, ys[0], yerr=stat, fmt='none', lw=.55, color=COLORS[name], zorder=4)
                bottom.errorbar(mids + .012, ys[0] - ref[0], yerr=stat, fmt='none', lw=.55, color=COLORS[name], zorder=4)
            top.stairs(ys[0], EDGES, baseline=None, color=COLORS[name], linestyle=STYLES[name], lw=1.15, zorder=3)
            bottom.stairs(ys[0] - ref[0], EDGES, baseline=None, color=COLORS[name], linestyle=STYLES[name], lw=1.15, zorder=3)
        top.set_title(r'$p_T^\ell>%d\,\mathrm{GeV}$' % cut, pad=5)
        top.set_ylabel(r'$A_\ell=(\sigma^+-\sigma^-)/(\sigma^++\sigma^-)$' if col == 0 else '')
        bottom.set_ylabel(r'$A_\ell-A_\ell^{\rm NNLO}$' if col == 0 else '')
        bottom.set_xlabel(r'$|\eta_\ell|$')
        bottom.axhline(0, color='#777777', lw=.5, zorder=0)
        top.margins(y=.16)
        bottom.margins(y=.17)
    labels = {'P': 'LO+PS shape with NNLO rates', 'B': 'Production moments',
              'A': '+ decay-angle moments', 'selected': 'Training-selected constraints'}
    joint_charges = [ch for ch in ['wp', 'wm'] if selected[ch] != 'A']
    if joint_charges == ['wm']:
        labels['selected'] = r'+ $W^-$ joint moments'
    elif joint_charges == ['wp']:
        labels['selected'] = r'+ $W^+$ joint moments'
    elif len(joint_charges) == 2:
        labels['selected'] = '+ joint moments'
    else:
        labels['selected'] = labels['A']
    handles = [Line2D([0], [0], color=COLORS[x], ls=STYLES[x], lw=1.2, label=labels[x]) for x in names]
    handles += [Line2D([0], [0], color='black', marker='o', ms=2.3, lw=0, label='Independent NNLO(W), MC errors'),
                Patch(facecolor='#777777', alpha=.22, label='Coherent seven-scale envelopes')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.015), ncol=2, frameon=False, columnspacing=2., handlelength=2.5)
    fig.subplots_adjust(top=.78, bottom=.12, left=.1, right=.985, hspace=.08, wspace=.25)
    for ext in ['pdf', 'png']:
        fig.savefig(out / ('w_independent_predictions.' + ext), dpi=220)
    plt.close(fig)
    caption = ('Independent predictions for the lepton charge asymmetry before and after tightening the lepton transverse-momentum cut. '
               'All other cuts remain pT_nu>25 GeV, mT>40 GeV and |eta_l|<2.5. No lepton distribution or tighter-cut rate entered constraint selection. '
               'Black points are a disjoint NNLO(W) integration; bars show channel-wise resampled Monte Carlo statistics. The separate red statistical bars combine training-integration and finite-prior statistics propagated through the constrained fit by linear response. '
               'Each prior charge is normalized to its training NNLO baseline rate, so the prior curve tests the shape information supplied by the shower. '
               'Grey and red shading show coherent seven-point scale envelopes for the reference and selected prediction, respectively; they are not combined with the statistical errors. '
               'Production moments constrain m_W, |y_W| and the resolved-recoil window; the angular candidate adds its fixed bulk window. '
               'The training rule selected W+: %s and W-: %s. Recoil information in the NNLO W input is NLO for W+jet. '
               'Lower panels show differences from the central independent reference. The same weights are used for both cuts.' % (selected['wp'], selected['wm']))
    (out / 'w_independent_predictions_caption.txt').write_text(caption + '\n')


def acceptance_table(data, out):
    reports = {ch: json.loads((data / ('w_%s_validation.json' % ch)).read_text()) for ch in ['wp', 'wm']}
    rows = []
    labels = {'P': 'LO+PS prior', 'B': 'Production moments', 'A': '+ decay-angle moments', 'selected': 'Training-selected model', 'reference': 'Independent NNLO'}
    for model in ['P', 'B', 'A', 'selected', 'reference']:
        cells = []
        plain = []
        for ch, cut in [('wp', '27'), ('wp', '30'), ('wm', '27'), ('wm', '30')]:
            name = reports[ch]['selected'] if model == 'selected' else ('B' if model == 'reference' else model)
            a = reports[ch]['models'][name]['acceptance'][cut]
            v = a['reference'] if model == 'reference' else a['value']
            e = a['reference_stat'] if model == 'reference' else a['stat']
            scales = np.array(a['reference_coherent_scales'] if model == 'reference' else a['coherent_scales'])
            up = 100 * (scales.max() - v)
            down = 100 * (v - scales.min())
            text = r'$%.2f\pm%.2f' % (100 * v, 100 * e)
            if model != 'P':
                text += r'{}^{+%.2f}_{-%.2f}' % (max(up, 0), max(down, 0))
            cells.append(text + '$')
            plain.append(dict(charge=ch, cut_GeV=int(cut), acceptance=v, stat=e, scale_up=up / 100, scale_down=down / 100))
        rows.append(dict(model=model, label=labels[model], cells=cells, values=plain))
    tex = ['% Acceptances in percent: value +- MC statistics, then seven-scale envelope.',
           r'\begin{tabular}{lcccc}', r'\toprule',
           r' & $W^+,\;27$ GeV & $W^+,\;30$ GeV & $W^-,\;27$ GeV & $W^-,\;30$ GeV \\', r'\midrule']
    tex += [row['label'] + ' & ' + ' & '.join(row['cells']) + r' \\' for row in rows]
    tex += [r'\bottomrule', r'\end{tabular}']
    (out / 'w_acceptance_table.tex').write_text('\n'.join(tex) + '\n')
    (out / 'w_acceptance_table.json').write_text(json.dumps(rows, indent=2) + '\n')


def all_scans(data, out):
    # Full model/cut scans are public supplementary plots, not automatic paper panels.
    for ch in ['wp', 'wm']:
        z = np.load(str(data / ('w_%s_validation.npz' % ch)), allow_pickle=False)
        meta = json.loads(str(z['metadata']))
        for tag in ['eta20', 'eta27', 'eta30', 'ptl', 'mt']:
            lo, hi = meta['slices'][tag]
            ix = np.arange(lo, hi)
            edges = np.array(meta['edges'][tag])
            ref = z['reference'][ix, 0]
            fig, ax = plt.subplots(figsize=(4.35, 2.8))
            mids = .5 * (edges[:-1] + edges[1:])
            err = np.sqrt(np.maximum(np.diag(z['reference_cov'])[ix], 0)) / np.abs(ref)
            ax.errorbar(mids, np.ones(len(ix)), yerr=err, fmt='o', ms=2, lw=.6, color='black', label='Independent NNLO(W)')
            labels = {'P': 'LO+PS (NNLO rate)', 'B': 'Production moments', 'A': '+ decay angle', 'J2': '+ joint degree 2', 'J4': '+ joint degree 4'}
            for model in ['P', 'B', 'A', 'J2', 'J4']:
                values = z[model + '_mean'][0, ix] / ref
                ax.stairs(values, edges, baseline=None, color=COLORS[model], ls=STYLES[model], lw=1., label=labels[model])
            ax.set_xlabel(r'$|\eta_\ell|$' if tag.startswith('eta') else (r'$p_T^\ell$ [GeV]' if tag == 'ptl' else r'$m_T$ [GeV]'))
            ax.set_ylabel('Prediction / independent NNLO')
            ax.set_title(ch.upper() + '  ' + tag)
            fig.legend(loc='upper center', bbox_to_anchor=(.5, 1.04), ncol=3, frameon=False)
            fig.subplots_adjust(top=.74, bottom=.17, left=.15, right=.98)
            fig.savefig(out / ('%s_%s_full_scan.pdf' % (ch, tag)))
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, nargs="?",
                        default=Path(__file__).resolve().parent / "data/w_joint")
    parser.add_argument("--output", type=Path, default=Path("w_joint_plots"))
    parser.add_argument("--full-scans", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    style()
    main_figure(args.data, args.output)
    acceptance_table(args.data, args.output)
    if __package__:
        from .plot_w_joint_map import plot_map
    else:
        from plot_w_joint_map import plot_map
    plot_map(args.data / "wm_joint_redistribution.npz", args.output)
    if args.full_scans:
        all_scans(args.data, args.output)
    print(f"Saved {args.output}. These plots read frozen comparisons; no new event fit runs here.")


if __name__ == "__main__":
    main()
