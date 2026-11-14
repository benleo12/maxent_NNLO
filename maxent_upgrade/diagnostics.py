"""Local event statistics and checks of joint moments outside a fit.

These quantify the supplied sample and targets. They cannot certify that an
unobserved region of phase space is adequately populated or modeled.
"""
import numpy as np

from .basis import cheb, umap
from .constraints import _window_fn


def _weights(weights):
    w = np.asarray(weights, float)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or (w < 0).any() or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("weights must be a nonempty, finite, nonnegative vector with positive sum")
    return w / w.sum()


def weight_diagnostics(weights, prior_weights=None):
    """Effective count and largest-event concentration, independent of units."""
    w = _weights(weights)
    square_sum = float(w @ w)
    out = dict(events=len(w), effective_events=1 / square_sum,
               largest_weight_fraction=float(w.max()),
               largest_variance_fraction=float(w.max()**2 / square_sum))
    if prior_weights is not None:
        p = _weights(prior_weights)
        if p.shape != w.shape:
            raise ValueError("prior and posterior weights must describe the same events")
        out["effective_fraction_retained"] = float((p @ p) / square_sum)
    return out


def bin_diagnostics(values, bins, weights, reference=None):
    """Counts, effective counts and event concentration for analysis bins.

    A supplied nonnegative reference is one value per bin, with any common
    normalization. `empty_with_reference` identifies bins with zero prior
    events and positive reference content. No inference is made without it.
    """
    values, bins = np.asarray(values, float), np.asarray(bins, float)
    w = _weights(weights)
    if values.shape != w.shape or not np.isfinite(values).all():
        raise ValueError("finite observable values must match the weight vector")
    if bins.ndim != 1 or len(bins) < 2 or not np.isfinite(bins).all() or not (np.diff(bins) > 0).all():
        raise ValueError("bins must be finite, strictly increasing edges")
    count = np.histogram(values, bins)[0]
    total = np.histogram(values, bins, weights=w)[0]
    squares = np.histogram(values, bins, weights=w * w)[0]
    index = np.searchsorted(bins, values, side="right") - 1
    index[values == bins[-1]] = len(bins) - 2
    inside = (index >= 0) & (index < len(bins) - 1)
    maximum = np.zeros(len(bins) - 1)
    np.maximum.at(maximum, index[inside], w[inside])
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(total), where=b > 0)
    out = dict(edges=bins, counts=count, weight_fraction=total,
               effective_events=divide(total**2, squares),
               largest_weight_fraction=divide(maximum, total),
               largest_variance_fraction=divide(maximum**2, squares))
    if reference is not None:
        ref = np.asarray(reference, float)
        if ref.shape != total.shape or not np.isfinite(ref).all() or (ref < 0).any():
            raise ValueError("reference must contain a finite nonnegative value per bin")
        out["empty_with_reference"] = (count == 0) & (ref > 0)
    return out


def joint_moment_residuals(events, weights, moments, config):
    """Evaluate supplied joint blocks without imposing them.

    Pass the configuration containing the joint blocks to check, even when the
    fit used only separate moments. Values are conditional on the product of
    the windows, matching the target's `values`. Window rates are reported
    separately. Residual/input_error is descriptive, not a joint significance
    test. Sample errors condition on the fitted weights and omit refit effects.
    """
    q = _weights(weights)
    out = {}
    for name, spec in (config.get("mixed") or {}).items():
        target = moments["mixed"][name]
        ox, oy = spec["observables"]
        x, y = np.asarray(events[ox], float), np.asarray(events[oy], float)
        if x.shape != q.shape or y.shape != q.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("joint observable arrays must be finite and match the weights")
        profiles = spec.get("profile") or {}
        w = _window_fn(x, profiles.get(ox)) * _window_fn(y, profiles.get(oy))
        qw = q * w
        rate = float(qw.sum())
        target_rate = float(target.get("rate", 1))
        if rate <= 0:
            out[name] = dict(available=False, reason="no posterior weight in joint window",
                             window_rate=rate, target_window_rate=target_rate, moments=[])
            continue
        a, b = spec["range"][ox]
        c, d = spec["range"][oy]
        maps = spec.get("map") or {}
        orders = list(target["values"])
        cx = cheb(umap(np.clip(x, a, b), a, b, maps.get(ox, "log")), max(k for k, n in orders))
        cy = cheb(umap(np.clip(y, c, d), c, d, maps.get(oy, "log")), max(n for k, n in orders))
        rows = []
        for k, n in orders:
            v = cx[:, k] * cy[:, n]
            actual = float(qw @ v / rate)
            expected = float(target["values"][k, n])
            error = float((target.get("errors") or {}).get((k, n), 0))
            stat = float((target.get("stat_errors") or {}).get((k, n), 0))
            scale = float((target.get("scale_errors") or {}).get((k, n), 0))
            if error <= 0:
                error = float(np.hypot(stat, scale))
            rows.append(dict(degrees=[int(k), int(n)], achieved=actual, target=expected,
                             residual=actual - expected, input_error=error if error > 0 else None,
                             residual_in_input_errors=(actual - expected) / error if error > 0 else None,
                             sample_error_fixed_weights=float(np.sqrt(np.sum(qw**2 * (v - actual)**2)) / rate)))
        out[name] = dict(available=True, window_rate=rate, target_window_rate=target_rate,
                         effective_events=float(rate**2 / (qw @ qw)), moments=rows)
    return out
