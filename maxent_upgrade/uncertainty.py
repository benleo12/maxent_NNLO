"""Re-solve variations with fixed moment orders and histogram their event weights.

`upgrade_with_uncertainties` supplies the variations in the released tables.
`upgrade_ensemble` accepts the same ingredients for any user-defined process.
Sources stay separate: envelopes, integration-run scatter and PDF Hessian errors
are different estimates and are not combined automatically.
"""
from dataclasses import dataclass, field

import numpy as np

from . import released
from .reweight import upgrade
from .provenance import array_identity, canonical_json, content_hash, fit_provenance, effective_configuration, released_table_identity


@dataclass
class Variation:
    """One shifted input. Omitted ingredients retain their central values.

    `prior_weight` is a replacement positive weight array on the SAME events.
    An explicit `config` changes maps/windows and starts from zero multipliers.
    `cross_section` is the total rate in pb, not an event-weight multiplier.
    """
    moments: object = None
    config: object = None
    prior_weight: object = None
    cross_section: object = None
    label: str = ""


@dataclass
class EnsembleResult:
    """Central diagnostics and unit-area weights for every variation."""
    central: object
    cross_section: float
    weights: dict
    cross_sections: dict
    methods: dict
    labels: dict
    notes: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def arrays(self, unit_area=False):
        """NPZ-ready arrays. By default all event weights are in pb.

        `weight_<source>` excludes the central result. PDF rows are eigenvector
        members 1..58; `weight_upgraded` supplies member 0 to the Hessian formula.
        Original event arrays can be saved alongside this dictionary.
        """
        out = dict(weight_upgraded=self.central.weights * (1 if unit_area else self.cross_section),
                   weight_normalization=np.asarray("unit_area" if unit_area else "pb"),
                   moment_names=np.asarray(list(self.central.chosen_moments), dtype=str),
                   moment_orders=np.asarray(list(self.central.chosen_moments.values()), dtype=int),
                   uncertainty_sources=np.asarray(list(self.weights), dtype=str),
                   uncertainty_methods=np.asarray(list(self.methods.values()), dtype=str),
                   uncertainty_notes=np.asarray(self.notes, dtype=str))
        if not unit_area:
            out["cross_section_upgraded_pb"] = np.asarray(self.cross_section)
        for source, w in self.weights.items():
            out["weight_" + source] = w if unit_area else w * self.cross_sections[source][:, None]
            out["labels_" + source] = np.asarray(self.labels[source], dtype=str)
            if not unit_area:
                out["cross_section_" + source + "_pb"] = self.cross_sections[source]
        if self.provenance:
            provenance = dict(self.provenance, normalization="unit_area" if unit_area else "pb",
                              output_weights=array_identity(out["weight_upgraded"]))
            out["provenance_json"] = np.asarray(canonical_json(provenance))
        return out

    def histogram(self, values, bins, *, unit_area=False, density=True):
        """Central histogram and separate lower/upper bands; see histogram_bands."""
        values, bins = _validate_histogram(values, bins)
        # Histogram one member at a time; do not allocate a second ensemble in pb.
        h = lambda w, rate: np.histogram(values, bins, weights=w * (1 if unit_area else rate))[0]
        central = h(self.central.weights, self.cross_section)
        members = {s: np.asarray([h(w, rate) for w, rate in zip(ws, self.cross_sections[s])])
                   for s, ws in self.weights.items()}
        out = _histogram_result(central, members, bins, self.methods, density)
        out["sample_error"] = _sample_error(values, bins, self.central.weights * self.cross_section,
                                            unit_area, density)
        return out


def _positive_rate(rate):
    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("cross sections must be finite positive numbers in pb")
    return rate


def upgrade_ensemble(events, moments, config, variations, *, cross_section=1.0,
                     methods=None, progress=None):
    """Upgrade any process and its input variations, keeping the central orders.

    `variations` maps source names to iterables of Variation objects. Use `scale`,
    `window` or `shower` for envelopes, `stat` for bootstrap replicas (at least 2),
    and `pdf` for CT18 eigenvector members 1..58 in pair order. Other sources use
    envelopes unless `methods` specifies `envelope`, `std` or `hessian68`.
    Iterables may be generators, so shifted moment tables can be loaded lazily.
    Stored weights take 8 * N_events * N_variations bytes, before file compression.
    `progress(label, result)` is called after each solve, including the central one.
    """
    rate = _positive_rate(cross_section)
    methods = dict(methods or {})
    chosen_methods = {}
    for source in variations:
        if not source.isidentifier() or source in {"upgraded", "target", "normalization"}:
            raise ValueError(f"invalid uncertainty source name {source!r}")
        method = methods.get(source, {"stat": "std", "pdf": "hessian68"}.get(source, "envelope"))
        if method not in {"envelope", "std", "hessian68"}:
            raise ValueError(f"unknown uncertainty method {method!r}")
        chosen_methods[source] = method
    central = upgrade(events, moments, config)
    provenance = fit_provenance(events, moments, config, central, cross_section=rate)
    provenance["variations"] = {}
    if progress:
        progress("central", central)
    orders = dict(central.chosen_moments)
    weights, rates, labels = {}, {}, {}
    for source, shifts in variations.items():
        ws, rs, names = [], [], []
        provenance["variations"][source] = []
        for i, shift in enumerate(shifts):
            shifted_events = events if shift.prior_weight is None else dict(events, weight=shift.prior_weight)
            shifted_config = dict(config if shift.config is None else shift.config)
            shifted_config.update(fixed_orders=orders, keep_features=False)
            # A moved measurement function must not inherit multipliers in the old basis.
            shifted_config.pop("lam0", None)
            if shift.config is None and shift.prior_weight is None:
                shifted_config["lam0"] = central.report["lam"]
            shifted_rate = _positive_rate(rate if shift.cross_section is None else shift.cross_section)
            shifted_moments = moments if shift.moments is None else shift.moments
            result = upgrade(shifted_events, shifted_moments, shifted_config)
            if result.chosen_moments != orders:
                raise ValueError(f"{source}: a variation changed the constrained moment orders")
            label = shift.label or f"{source} {i + 1}"
            ws.append(result.weights)
            rs.append(shifted_rate)
            names.append(label)
            provenance["variations"][source].append(dict(
                label=label, moments_sha256=content_hash(shifted_moments),
                configuration=effective_configuration(shifted_config), cross_section_pb=shifted_rate,
                shape_weights=array_identity(result.weights),
                prior_weights=(array_identity(shift.prior_weight) if shift.prior_weight is not None else None)))
            if progress:
                progress(label, result)
        _check_count(source, chosen_methods[source], len(ws))
        weights[source], rates[source], labels[source] = np.asarray(ws), np.asarray(rs), names
    return EnsembleResult(central, rate, weights, rates, chosen_methods, labels, provenance=provenance)


def _check_count(source, method, n):
    if n == 0 or (method == "std" and n < 2) or (method == "hessian68" and n % 2):
        raise ValueError(f"{source}: need nonempty variations, at least two for std, and paired members for Hessian errors")


def upgrade_with_uncertainties(events, process, *, sources=("scale", "stat", "window", "pdf"),
                               n_boot=20, seed=1, prior_weights=None, progress=None, **selection):
    """Automate the released scale, integration-statistical, window and PDF inputs.

    The baseline must use the nominal window. Missing window/PDF inputs are listed
    in `result.notes`, not interpreted as zero uncertainty. `prior_weights` may map
    shower-variation labels to replacement positive weights on the same events.
    It adds a separate shower envelope with the central fixed-order rate.
    `selection` accepts the born/recoil/mixed/max_orders/nnlo_zj/qt_basis loader options.
    """
    sources = tuple(sources)
    if len(set(sources)) != len(sources) or set(sources) - {"scale", "stat", "window", "pdf"}:
        raise ValueError("sources must be distinct entries from scale, stat, window, pdf")
    if selection.get("window", "nominal") not in (None, "nominal"):
        raise ValueError("uncertainty ensembles require a nominal central window")
    for key in ("scale", "runs", "pdf_member"):
        if key in selection:
            raise ValueError(f"{key} is selected by the uncertainty workflow")
    if "stat" in sources and (isinstance(n_boot, bool) or int(n_boot) != n_boot or n_boot < 2):
        raise ValueError("statistical bands require n_boot >= 2 bootstrap replicas")
    selection = dict(selection)
    selection.pop("window", None)
    moments, config = released.load_moments(process, **selection)
    table = released.table(process)
    variations, notes = {}, []

    def shifted(**kwargs):
        m, c = released.load_moments(process, **selection, **kwargs)
        changes_basis = "window" in kwargs
        rate_args = {k: v for k, v in kwargs.items() if k in ("scale", "runs", "pdf_member")}
        return Variation(moments=m, config=c if changes_basis else None,
                         cross_section=released.cross_section(process, **rate_args),
                         label=next((f"{k} {v}" for k, v in kwargs.items() if k != "check_seam"), ""))

    if "scale" in sources:
        variations["scale"] = (shifted(scale=k, check_seam=False) for k in range(1, len(table["scales"])))
    if "stat" in sources:
        n_runs = int(table["n_independent_runs"])
        if n_runs < 2:
            raise ValueError("bootstrap propagation requires at least two independent fixed-order runs")

        def replicas():
            rng = np.random.default_rng(seed)
            for k in range(int(n_boot)):
                v = shifted(runs=rng.integers(0, n_runs, n_runs), check_seam=False)
                v.label = f"integration replica {k + 1}"
                yield v
        variations["stat"] = replicas()
        if selection.get("nnlo_zj"):
            notes.append("NNLO Z+jet has no per-run replicas: stat resamples only the Drell-Yan input.")
    if "window" in sources:
        available = released.windows(process)
        selected_recoil = table["default_constraints"]["recoil"] if selection.get("recoil") is None else selection["recoil"]
        movable = (set(released._window(process, "down")[0]) if available else set())
        if available and movable.intersection(selected_recoil) and not selection.get("nnlo_zj"):
            variations["window"] = (shifted(window=w) for w in ("down", "up"))
        else:
            notes.append("No moved-window inputs apply to the selected constraints.")
    if "pdf" in sources:
        pdf = released.pdf_table(process)
        if pdf is not None and not selection.get("nnlo_zj"):
            variations["pdf"] = (shifted(pdf_member=k) for k in range(1, int(pdf["n_members"])))
        else:
            notes.append("No PDF-member inputs are released for the selected calculation.")
    if prior_weights:
        variations["shower"] = (Variation(prior_weight=w, label=label) for label, w in prior_weights.items())
    else:
        notes.append("Shower variations require generator-supplied prior weights or separate samples.")
    result = upgrade_ensemble(events, moments, config, variations,
                              cross_section=released.cross_section(process), progress=progress)
    result.provenance["moment_table"] = released_table_identity(process, table, selection, sources)
    result.provenance["uncertainty_workflow"] = dict(sources=list(sources), bootstrap_replicas=int(n_boot), bootstrap_seed=seed)
    result.notes.extend(notes)
    return result


def _histogram_result(central, members, edges, methods, density):
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or len(edges) != len(central) + 1 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError("bins must be finite, strictly increasing edges")
    widths = np.diff(edges) if density else np.ones(len(central))
    out = dict(edges=edges, central=central / widths)
    for source, h in members.items():
        method = methods[source]
        _check_count(source, method, len(h))
        if method == "envelope":
            low, high = np.minimum(h.min(axis=0), central), np.maximum(h.max(axis=0), central)
        elif method == "std":
            error = h.std(axis=0, ddof=1)
            low, high = central - error, central + error
        elif method == "hessian68":
            up, down = released.hessian_band(np.vstack([central, h]))
            low, high = central - down, central + up
        else:
            raise ValueError(f"unknown uncertainty method {method!r}")
        out[source + "_low"], out[source + "_high"] = low / widths, high / widths
    return out


def _validate_histogram(values, bins):
    values, bins = np.asarray(values), np.asarray(bins, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("observable values must be finite and one-dimensional")
    if bins.ndim != 1 or len(bins) < 2 or not np.isfinite(bins).all() or not (np.diff(bins) > 0).all():
        raise ValueError("bins must be finite, strictly increasing edges")
    return values, bins


def _sample_error(values, bins, weights, unit_area, density):
    # Usual weighted-event error, conditional on the fitted weights. For a shape,
    # account for the covariance with its normalization (including under/overflow).
    w = weights / weights.sum() if unit_area else weights
    h2 = np.histogram(values, bins, weights=w * w)[0]
    if unit_area:
        h = np.histogram(values, bins, weights=w)[0]
        h2 = (1 - 2 * h) * h2 + h * h * np.sum(w * w)
    return np.sqrt(np.maximum(h2, 0)) / (np.diff(bins) if density else 1)


def histogram_bands(values, bins, saved, *, unit_area=False, density=True):
    """Histogram an NPZ output (or dictionary) from the uncertainty workflow.

    Returns edges, central, sample_error and <source>_low/high arrays. The sample
    error treats the fitted event weights as fixed, with normalization covariance
    for unit-area shapes. It is separate from the fixed-order integration errors.
    Default units are pb
    per unit of the observable when the saved weights are in pb. `unit_area`
    divides each member by its FULL event-weight sum, including events outside
    the bins. `density` divides only by bin width. Never histogram up/down event
    weight envelopes: coherent variations must be histogrammed first.
    """
    values, bins = _validate_histogram(values, bins)
    unit_area = unit_area or str(saved.get("weight_normalization", "pb")) == "unit_area"

    def hist(w):
        w = np.asarray(w)
        if w.shape != values.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("event weights must match the observable and have a finite positive total")
        return np.histogram(values, bins, weights=w / w.sum() if unit_area else w)[0]
    central = hist(saved["weight_upgraded"])
    sources = list(saved.get("uncertainty_sources", []))
    methods = list(saved.get("uncertainty_methods", []))
    if len(sources) != len(methods):
        raise ValueError("uncertainty source/method metadata lengths differ")
    members = {s: np.asarray([hist(w) for w in saved["weight_" + s]]) for s in sources}
    out = _histogram_result(central, members, bins, dict(zip(sources, methods)), density)
    out["sample_error"] = _sample_error(values, bins, np.asarray(saved["weight_upgraded"]), unit_area, density)
    return out
