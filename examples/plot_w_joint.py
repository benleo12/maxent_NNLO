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


def envelope(ax, edges, scales, color, alpha=.15):
    lo = scales.min(0)
    hi = scales.max(0)
    ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step='post', color=color, alpha=alpha, linewidth=0)


def main_figure(data, out):
    asym = json.loads((data / 'charge_asymmetry.json').read_text())
    selected = {ch: json.loads((data / ('w_%s_validation.json' % ch)).read_text())['selected'] for ch in ['wp', 'wm']}
    names = ['P', 'B', 'A', 'selected'] if any(x != 'A' for x in selected.values()) else ['P', 'B', 'selected']
    source = Path(__file__).resolve().parent / 'data/paper_distributions/w/charge_asymmetry.npz'
    with np.load(source, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['eta_edges'], EDGES)
        measured = archive['atlas_born_asymmetry']
        measured_error = np.vstack((archive['atlas_total_error_low'], archive['atlas_total_error_high']))

    fig, axs = plt.subplots(3, 2, figsize=(7.1, 5.8), sharex='col',
                            gridspec_kw={'height_ratios': [1.7, 1., 1.]})
    mids = .5 * (EDGES[1:] + EDGES[:-1])
    for col, cut in enumerate((20, 30)):
        tag = f'eta{cut}'
        top, fo, atlas = axs[:, col]
        ref = np.asarray(asym['reference'][tag]['coherent_scales'])
        rs = np.asarray(asym['reference'][tag]['stat'])
        if not np.all(np.isfinite(ref) & (ref > 0)):
            raise ValueError(f'Invalid NNLO asymmetry for {cut} GeV')
        envelope(top, EDGES, ref, '#777777', .22)
        envelope(fo, EDGES, ref / ref[0], '#777777', .22)
        for ax, values, error in ((top, ref[0], rs), (fo, np.ones_like(rs), rs / ref[0])):
            ax.errorbar(mids, values, yerr=error, fmt='s', mfc='white', ms=2.8,
                        lw=.65, color='black', zorder=5)
        panels = [(top, 1.), (fo, ref)]
        if cut == 20:
            panels.append((atlas, measured))
            top.errorbar(mids, measured, yerr=measured_error, fmt='o', ms=2.5,
                         lw=.65, color='black', zorder=6)
            low, high = measured_error / measured
            atlas.stairs(1 + high, EDGES, baseline=1 - low, fill=True, color='.88', zorder=0)
            atlas.axhline(1, color='.45', lw=.55)
            atlas.set(xlabel=r'$|\eta_\ell|$', ylabel='Ratio to ATLAS')
        else:
            atlas.axis('off')
            atlas.text(.5, .5, 'ATLAS measured the 20 GeV selection.',
                       ha='center', va='center', transform=atlas.transAxes)
            fo.tick_params(labelbottom=True)
            fo.set_xlabel(r'$|\eta_\ell|$')
        for name in names:
            ys = np.asarray(asym[name][tag]['coherent_scales'])
            stat = np.asarray(asym[name][tag]['stat'])
            for ax, denominator in panels:
                values = ys / denominator
                ax.stairs(values[0], EDGES, baseline=None, color=COLORS[name],
                          linestyle=STYLES[name], lw=1.15, zorder=3)
                if name == 'selected':
                    envelope(ax, EDGES, values, COLORS[name], .16)
                    central_denominator = ref[0] if ax is fo else denominator
                    ax.errorbar(mids + .012, values[0], yerr=stat / central_denominator,
                                fmt='none', lw=.55, color=COLORS[name], zorder=4)
        for ax, _ in panels:
            ax.set_xlim(0, 2.5)
            ax.tick_params(which='both', length=3)
            ax.margins(y=.12)
        fo.axhline(1, color='#777777', lw=.5, zorder=0)
        top.set_title(r'$p_T^\ell>%d\,\mathrm{GeV}$' % cut, pad=5)
        if col == 0:
            top.set_ylabel(r'$A_\ell$')
            fo.set_ylabel('Ratio to NNLO')
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
    handles += [Line2D([], [], color='black', marker='s', mfc='white', ms=2.8, lw=0, label='Independent NNLO'),
                Line2D([], [], color='black', marker='o', ms=2.5, lw=0, label='ATLAS 7 TeV'),
                Patch(facecolor='#777777', alpha=.22, label='Seven-point scale variation')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               columnspacing=1.5, handlelength=2.)
    fig.subplots_adjust(top=.845, bottom=.09, left=.11, right=.98, hspace=.25, wspace=.25)
    for ext in ['pdf', 'png']:
        fig.savefig(out / ('w_independent_predictions.' + ext), dpi=220)
    plt.close(fig)
    caption = ('Lepton charge asymmetry at 7 TeV for two lepton transverse-momentum cuts. '
               'All curves use the same event weights. The missing transverse momentum exceeds '
               '25 GeV and the transverse mass exceeds 40 GeV. Open squares show an independent '
               'NNLO calculation. The middle panels compare each prediction with NNLO at the same '
               'renormalization and factorization scales. The bands span seven scale choices. '
               'Bars on the NNLO and selected predictions give their separate numerical errors. '
               'Filled circles and the lower panel compare with the ATLAS measurement at 20 GeV '
               '(arXiv:1109.5141, Table XXVI). The grey band there gives the experimental uncertainty.')
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
    path = Path(__file__).resolve().parent / "data/w_joint/atlas_charge_eta20.csv"
    measured = np.genfromtxt(path, names=True, delimiter=",")
    np.testing.assert_array_equal(np.r_[measured["abs_eta_low"], measured["abs_eta_high"][-1]], EDGES)
    # Full model/cut scans are public supplementary plots, not automatic paper panels.
    for ch in ['wp', 'wm']:
        z = np.load(str(data / ('w_%s_validation.npz' % ch)), allow_pickle=False)
        meta = json.loads(str(z['metadata']))
        for tag in ['eta20', 'eta27', 'eta30', 'ptl', 'mt']:
            lo, hi = meta['slices'][tag]
            ix = np.arange(lo, hi)
            edges = np.array(meta['edges'][tag])
            ref = z['reference'][ix, 0]
            if tag == 'eta20':
                fig, (ax, atlas) = plt.subplots(2, 1, sharex=True, figsize=(5.5, 4.6))
                # The paper tabulates dσ/dη per signed hemisphere. Our bins sum both signs.
                rate = 2 * np.diff(edges) * measured[ch + '_pb_per_signed_eta']
                error = np.hypot(measured[ch + '_error_percent'] / 100, .034)
                atlas.stairs(1 + error, edges, baseline=1 - error, fill=True, color='.88')
                atlas.axhline(1, color='.45', lw=.55)
                atlas.set_ylabel('Prediction / ATLAS')
                xlabel_axis = atlas
            else:
                fig, ax = plt.subplots(figsize=(5.5, 2.8))
                xlabel_axis = ax
            mids = .5 * (edges[:-1] + edges[1:])
            err = np.sqrt(np.maximum(np.diag(z['reference_cov'])[ix], 0)) / np.abs(ref)
            ax.errorbar(mids, np.ones(len(ix)), yerr=err, fmt='o', ms=2, lw=.6, color='black', label='Independent NNLO(W)')
            labels = {'P': 'LO+PS (NNLO rate)', 'B': 'Production moments', 'A': '+ decay angle', 'J2': '+ joint degree 2', 'J4': '+ joint degree 4'}
            for model in ['P', 'B', 'A', 'J2', 'J4']:
                prediction = z[model + '_mean'][0, ix]
                ax.stairs(prediction / ref, edges, baseline=None, color=COLORS[model],
                          ls=STYLES[model], lw=1., label=labels[model])
                if tag == 'eta20':
                    atlas.stairs(prediction / rate, edges, baseline=None,
                                 color=COLORS[model], ls=STYLES[model], lw=1.)
            xlabel_axis.set_xlabel(r'$|\eta_\ell|$' if tag.startswith('eta') else (r'$p_T^\ell$ [GeV]' if tag == 'ptl' else r'$m_T$ [GeV]'))
            ax.set_ylabel('Prediction / independent NNLO')
            ax.set_title(ch.upper() + '  ' + tag)
            fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
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
