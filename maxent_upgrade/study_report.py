"""Small standalone figures and a local HTML report for a frozen study.

Matplotlib is optional and imported only when a report is requested. Numerical
results and provenance are always written alongside the figures as JSON/NPZ.
"""
from html import escape
from pathlib import Path

import numpy as np

from .provenance import canonical_json


COLORS = ("#87939d", "#cf7a2b", "#23658b", "#8871a7", "#278775")
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
         "axes.titlesize": 12, "legend.fontsize": 8.5, "xtick.labelsize": 9,
         "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": .7, "lines.linewidth": 1.6, "savefig.bbox": "tight",
         "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42}


def write_study_report(directory, *, title, models, diagnostics, validations,
                       edges, xlabel, selected, anchor, comparison=None, provenance=None):
    """Write training.pdf/png, predictions.pdf/png, report.html/json and arrays.npz.

    ``models`` maps display names to ConstraintFit objects. ``diagnostics`` maps
    each model to a mapping of block names to ``fit.diagnose`` results.
    ``validations`` maps models to ``validate_prediction`` results for one common
    binned observable. Predictions are *bin integrals* in pb; ``edges`` control
    display only, and must already occur in the frozen observable definition.
    Additional observables can use another report directory, without refitting.
    All candidates remain in the data/HTML. The spectrum defaults to the prior
    (if present), anchor and selected model; ``comparison`` can name a small
    alternative display set. Training diagnostics paginate at sixteen rows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(models)
    if set(names) != set(validations) or selected not in names or anchor not in names:
        raise ValueError("report every model and name an available selected model and anchor")
    prior = next((name for name, fit in models.items() if not fit.labels), None)
    comparison = list(dict.fromkeys(comparison or [n for n in (prior, anchor, selected) if n]))
    if any(n not in names for n in comparison) or selected not in comparison or len(comparison) > 6:
        raise ValueError("the displayed comparison must include the selected model and at most six available candidates")
    colors = {name: COLORS[(i + 1) % len(COLORS)] for i, name in enumerate(names)}
    if prior:
        colors[prior] = "#87939d"
    colors[anchor], colors[selected] = "#23658b", "#bd3c3c"
    edges = np.asarray(edges, float)
    nbin = len(validations[selected]["prediction"].labels)
    if edges.shape != (nbin + 1,) or not np.all(np.diff(edges) > 0):
        raise ValueError("strictly increasing bin edges must match the held-out predictions")
    for result in validations.values():
        definition = result["definition"]
        if not isinstance(definition, dict) or definition.get("edges") != edges.tolist():
            raise ValueError("display edges must equal the frozen validation definition's edges")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays, record = {}, dict(title=title, selected=selected, anchor=anchor, models={},
                              edges=edges.tolist(), xlabel=xlabel, comparison=comparison,
                              colors=colors, provenance=provenance or {},
                              interpretation="Input agreement, training correlation evidence and independent prediction are distinct checks. None certifies accuracy for other observables.")
    rows, diagnostic_rows = [], []
    for name, fit in models.items():
        covariance = fit.central.report["target_covariance"]
        error = np.sqrt(np.maximum(np.diag(covariance), 0))
        residual = fit.central.report["constraint_residuals"]
        pulls = np.divide(np.abs(residual), error, out=np.zeros_like(residual), where=error > 0)
        exact = float(np.max(np.abs(residual[:, error == 0]), initial=0))
        worst = float(np.max(pulls, initial=0)) if np.any(error > 0) and exact <= 1e-12 else None
        retained = fit.central.effN_retained
        rows.append((name, worst, retained, bool(fit.labels)))
        entry = dict(input_max_pull=worst, zero_error_max_absolute_residual=exact,
                     effective_events_retained=retained,
                     freeze_id=fit.frozen_id, diagnostics={})
        for block, result in diagnostics.get(name, {}).items():
            entry["diagnostics"][block] = dict(test=result["test"],
                                               residual=result["residual"].summary(),
                                               omitted_fitted_labels=result["omitted_fitted_labels"])
            diagnostic_rows.append((name, block, result["test"]))
        for kind in ("prediction", "reference", "difference"):
            prediction = validations[name][kind]
            entry[kind] = prediction.summary()
            _save_arrays(arrays, f"model_{names.index(name)}_{kind}", prediction)
        change = validations[name]["prediction"].difference(validations[anchor]["prediction"])
        entry["paired_change_from_anchor"] = change.summary()
        _save_arrays(arrays, f"model_{names.index(name)}_paired_change", change)
        record["models"][name] = entry
    arrays["edges"] = edges
    arrays["metadata"] = np.array(canonical_json(record))
    np.savez_compressed(directory / "arrays.npz", **arrays)
    (directory / "report.json").write_text(canonical_json(record) + "\n")
    with plt.rc_context(STYLE):
        note = (provenance or {}).get("display_note", "")
        pages = _training_figure(plt, directory, rows, diagnostic_rows, colors, note)
        _prediction_figure(plt, directory, comparison, validations, edges, xlabel, selected, anchor, colors, note)
    (directory / "report.html").write_text(_html(title, rows, diagnostic_rows, selected, anchor, pages))
    return record


def _save_arrays(arrays, prefix, prediction):
    arrays[prefix + "_mean"] = prediction.mean
    arrays[prefix + "_cov_numerical"] = prediction.covariance_numerical
    arrays[prefix + "_cov_prior"] = prediction.covariance_prior
    for i, source in enumerate(prediction.sources.values()):
        arrays[f"{prefix}_source_{i}_jacobian"] = source["jacobian"]
        arrays[f"{prefix}_source_{i}_covariance"] = source["covariance"]


def _save(plt, fig, directory, name):
    fig.savefig(directory / (name + ".pdf"))
    fig.savefig(directory / (name + ".png"), dpi=180)
    plt.close(fig)


def _training_figure(plt, directory, rows, diagnostics, colors, note):
    pages = []
    for start in range(0, max(len(diagnostics), 1), 16):
        shown = diagnostics[start:start + 16]
        first = start == 0
        heights = ([max(1.6, .28 * len(rows))] if first else []) + [max(1.8, .28 * len(shown))]
        fig, axes = plt.subplots(len(heights), 1, figsize=(7.2, sum(heights) + 1.6),
                                 squeeze=False, constrained_layout=True,
                                 gridspec_kw=dict(height_ratios=heights))
        axes = axes[:, 0]
        if note:
            fig.suptitle(note, fontsize=10, color=".4")
        if first:
            _closure_panel(axes[0], rows, colors)
        _diagnostic_panel(axes[-1], shown, colors)
        name = "training" if first else f"training_{1 + start // 16}"
        _save(plt, fig, directory, name)
        pages.append(name)
    return pages


def _closure_panel(ax, rows, colors):
    for i, (name, pull, _, imposed) in enumerate(rows):
        if pull is not None:
            ax.plot(pull, i, "o", color=colors[name], ms=7)
        else:
            ax.text(.02, i, "MC pull unavailable" if imposed else "No imposed moments",
                    va="center", color=COLORS[0], fontsize=8)
    ax.axvline(1, color=".5", ls="--", lw=1)
    ax.set(yticks=range(len(rows)), yticklabels=[r[0] for r in rows],
           xlabel="Largest imposed-moment residual / input MC error",
           title="1  Agreement with the supplied inputs")
    ax.set_xlim(-.03, max(1.15, max((r[1] or 0 for r in rows), default=0) * 1.1))
    ax.set_ylim(len(rows) - .5, -.5)
    ax.grid(axis="x", alpha=.15)


def _diagnostic_panel(ax, diagnostics, colors):
    labels = []
    for i, (name, block, test) in enumerate(diagnostics):
        labels.append(f"{name} · {block}")
        pvalues = test.get("p_values")
        if pvalues is None or test["status"] in {"discarded_modes_require_review", "no_statistical_rank"}:
            label = ("no unfitted columns" if test["status"] == "no_unfitted_constraints"
                     else "covariance modes need review")
            ax.text(.12, i, label, va="center", fontsize=8, color=".45")
            continue
        score = -np.log10(np.maximum(pvalues, 1e-12))
        color = colors[name]
        ax.plot([score.min(), score.max()], [i, i], color=color, lw=3, alpha=.35)
        ax.plot(score.min(), i, "o", color=color, ms=6)
    for threshold in {test.get("significance", .01) for _, _, test in diagnostics}:
        ax.axvline(-np.log10(threshold), color=".5", ls="--", lw=1)
    ax.set(yticks=range(len(labels)), yticklabels=labels,
           xlabel=r"Residual evidence  $-\log_{10}p$  (range over coherent inputs)",
           title="2  Information left in the candidate correlations", xlim=(-.1, 12.5))
    ax.set_ylim(max(len(labels) - .5, .5), -.5)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12], ["0", "2", "4", "6", "8", "10", "≥12"])
    ax.grid(axis="x", alpha=.15)


def _prediction_figure(plt, directory, names, validations, edges, xlabel, selected, anchor, colors, note):
    centers, widths = (edges[1:] + edges[:-1]) / 2, np.diff(edges)
    reference = validations[selected]["reference"]
    ref = reference.mean[0]
    referr = np.sqrt(np.maximum(np.diag(reference.covariance), 0))
    valid = np.all(reference.mean > 0, axis=0)
    chosen = validations[selected]["prediction"]
    color = colors[selected]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.1), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=(2.3, 1, 1)))
    if note:
        fig.suptitle(note, fontsize=10, color=".4")
    for i, name in enumerate(names):
        prediction = validations[name]["prediction"]
        if not np.array_equal(validations[name]["reference"].mean, reference.mean):
            raise ValueError("all candidate predictions must use the same independent reference")
        axes[0].stairs(prediction.mean[0] / widths, edges, baseline=None,
                       color=colors[name], label=name)
        ratio = np.divide(prediction.mean[0], ref, out=np.full(len(ref), np.nan), where=valid)
        axes[1].stairs(ratio, edges, baseline=None, color=colors[name])
    low, high = chosen.mean.min(axis=0), chosen.mean.max(axis=0)
    axes[0].stairs(high / widths, edges, baseline=low / widths, fill=True,
                   color=color, alpha=.13, label="Selected: coherent input variations")
    error = np.sqrt(np.maximum(np.diag(chosen.covariance), 0))
    axes[0].errorbar(centers, chosen.mean[0] / widths, yerr=error / widths,
                     fmt="none", ecolor=color, capsize=2, lw=1,
                     label="Selected: target + refitted-prior MC")
    axes[0].errorbar(centers, ref / widths, yerr=referr / widths,
                     fmt=".", color="#252c34", ms=4, capsize=2, lw=1,
                     label="Independent reference: MC")
    ylabel = validations[selected]["definition"].get("density_label", "Cross section / bin width")
    axes[0].set(ylabel=ylabel, title="3  Predictions after freezing the fit")
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.10), ncol=2, frameon=False)
    scale_ratios = np.divide(chosen.mean, reference.mean,
                             out=np.full_like(chosen.mean, np.nan), where=valid[None, :])
    axes[1].stairs(scale_ratios.max(axis=0), edges, baseline=scale_ratios.min(axis=0),
                   fill=True, color=color, alpha=.13)
    relerr = np.divide(referr, ref, out=np.full_like(ref, np.nan), where=valid)
    axes[1].stairs(1 + relerr, edges, baseline=1 - relerr, fill=True, color=".5", alpha=.16)
    axes[1].errorbar(centers[valid], chosen.mean[0, valid] / ref[valid],
                     yerr=error[valid] / ref[valid], fmt="none", ecolor=color, capsize=2, lw=1)
    axes[1].axhline(1, color=".5", lw=.7)
    axes[1].set(ylabel="/ reference")
    change = chosen.difference(validations[anchor]["prediction"])
    scale = np.divide(100., ref, out=np.full_like(ref, np.nan), where=valid)
    axes[2].stairs(change.mean.max(axis=0) * scale, edges,
                   baseline=change.mean.min(axis=0) * scale, fill=True, color=color, alpha=.13)
    axes[2].errorbar(centers, change.mean[0] * scale,
                     yerr=np.sqrt(np.maximum(np.diag(change.covariance), 0)) * scale,
                     fmt="o", color=color, ms=3, capsize=2, lw=1)
    axes[2].axhline(0, color=".5", lw=.7)
    axes[2].set(ylabel="Fit change\n[% of reference]", xlabel=xlabel)
    axes[2].text(.02, .94, f"{selected} − {anchor}", transform=axes[2].transAxes,
                 ha="left", va="top", fontsize=8, color=color)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_xlim(edges[0], edges[-1])
    _save(plt, fig, directory, "predictions")


def _html(title, rows, diagnostics, selected, anchor, pages):
    table = "".join(f"<tr><td>{escape(name)}</td><td>{'—' if pull is None else f'{pull:.3g}'}</td>"
                    f"<td>{100 * retained:.2f}%</td></tr>" for name, pull, retained, _ in rows)
    tests = "".join(f"<tr><td>{escape(name)}</td><td>{escape(block)}</td>"
                    f"<td>{test['rank']}/{test['dimension']}</td>"
                    f"<td>{escape(test['status'].replace('_', ' '))}</td></tr>"
                    for name, block, test in diagnostics)
    training_figures = "".join(f'<img src="{name}.png" alt="Training diagnostics">'
                               f'<p><a href="{name}.pdf">Training figure (PDF)</a></p>' for name in pages)
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#243340;background:#f4f7f9;margin:0}}
main{{max-width:980px;margin:40px auto;background:white;padding:36px 48px;border-radius:12px}}
h1{{font-size:30px;line-height:1.2}}h2{{margin-top:36px;font-size:22px}}p{{max-width:80ch}}
.tag{{color:#23658b;font-size:13px;letter-spacing:.08em;text-transform:uppercase}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #dce4ea}}
th{{color:#526475}}img{{width:100%;height:auto}}a{{color:#23658b}}.note{{background:#edf3f7;padding:14px 18px;border-left:3px solid #23658b}}
@media(max-width:700px){{main{{margin:0;padding:24px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}}}</style>
<main><div class="tag">Frozen constraint study</div><h1>{escape(title)}</h1>
<p>Three questions guide this report: does the fit reproduce its inputs, do supplied
correlations reveal missing information, and how well does the frozen fit predict an
independent calculation?</p><p class="note">Selected candidate: <b>{escape(selected)}</b>.
Model changes are measured relative to <b>{escape(anchor)}</b>. These checks concern the
declared functions and predictions; they do not certify accuracy for other observables.</p>
<h2>Training and the decision to add information</h2>
<table><tr><th>Candidate</th><th>Largest input residual / MC error</th><th>Effective events retained</th></tr>{table}</table>
<p>Input residuals include every coherent scale. A correlation test uses the full
common-run covariance and the response of the fitted weights. Its scale variations
remain separate alternatives. A block with no unfitted columns supplies no further test.</p>
<table><tr><th>Candidate</th><th>Diagnostic block</th><th>Retained rank</th><th>Outcome</th></tr>{tests}</table>
{training_figures}
<h2>Independent predictions</h2><img src="predictions.png" alt="Held-out spectrum, reference ratios and paired model changes">
<p>Filled bands show coherent scale alternatives. Error bars on the selected model
combine target-integration and refitted-prior sampling errors at first order; their
separate covariance matrices are saved. Reference errors are shown separately. The
ratio band pairs each prediction scale with the same reference scale. The lower panel
uses common-input and common-event covariance for the model difference, divided by
the central reference solely for display.</p>
<p><a href="predictions.pdf">Prediction figure (PDF)</a> ·
<a href="report.json">Results and provenance (JSON)</a> · <a href="arrays.npz">Numerical arrays (NPZ)</a></p>
<p>Prior errors assume independent weighted event rows. Scale bands are not combined
with Monte Carlo covariance. A frozen record checks unchanged declared choices and
available run identities; it cannot establish an analyst's prior knowledge of a reference.</p></main></html>"""
