r"""The fixed-order moments released with the paper, as the two inputs of upgrade().

    from maxent_upgrade import upgrade, load_moments
    moments, config = load_moments("dy")          # the paper's constraint set for Drell-Yan
    result = upgrade(events, moments, config)    # events: your prior, see moments/README.md

Processes: "dy" (Z/gamma* -> l+l-, NNLO), "wp", "wm" (W+, W-, NNLO), "aa" (diphoton, NNLO),
"ggh" (gluon-fusion Higgs, NNLO, inclusive), "zj" (Z+jet, NLO).  Each table is a JSON file
in moments/ (installed as maxent_upgrade/moments/); <name>_runs.json.gz next to it holds the
channel-summed accumulators of every independent run at all seven scales, from which
`scale=` and `runs=` pool the moments again (the uncertainty bands of the paper).
"""
import gzip
import json
import os

import numpy as np

PROCESSES = {"dy": "dy_nnlo_moments", "wp": "w_nnlo_moments_WP", "wm": "w_nnlo_moments_WM",
             "aa": "aa_nnlo_moments", "ggh": "ggh_nnlo_moments", "zj": "zj_nlo_moments"}

_RUNS_CACHE = {}


def _data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(here, "moments"), os.path.join(os.path.dirname(here), "moments")):
        if os.path.isfile(os.path.join(d, PROCESSES["dy"] + ".json")):
            return d
    raise FileNotFoundError("released moment tables not found: reinstall the package "
                            "(pip install .) or run from a clone of the repository")


def _path(process, suffix=".json"):
    name = PROCESSES.get(process, process)
    if name.endswith(".json"):
        return name
    return os.path.join(_data_dir(), name + suffix)


def processes():
    """The process keys of the released tables."""
    return list(PROCESSES)


def table(process):
    """The raw JSON table of a process (a key of PROCESSES, or a path to a .json file)."""
    with open(_path(process)) as f:
        return json.load(f)


def observables(process):
    """{events key: description} for every tower the table carries."""
    return {d["observable"]: d["note"] for d in table(process)["towers"].values()}


def sigma_fid(process):
    """The fixed-order fiducial cross section {value, stat, scale_lo, scale_hi} in pb, or None
    where the production's normalization is not validated (the Higgs table)."""
    return table(process)["sigma_fid_pb"]


def _runs(process):
    """The per-run accumulators: (tags -> column, acc[run, tag, scale])."""
    if process not in _RUNS_CACHE:
        with gzip.open(_path(process, "_runs.json.gz"), "rt") as f:
            r = json.load(f)
        _RUNS_CACHE[process] = ({t: i for i, t in enumerate(r["tags"])}, np.asarray(r["acc"], float))
    return _RUNS_CACHE[process]


def _repool(process, num_tags, den_tag, scale, runs):
    """Values, stat and scale errors of ratio-of-sums moments, pooled over `runs` (positions,
    repeats allowed) at `scale`, exactly as export_moments.py pools the released numbers."""
    col, A = _runs(process)
    A = A[np.asarray(runs, int)] if runs is not None else A
    den = A[:, col[den_tag], :]                                  # (run, scale)
    vals, stat, scl = [], [], []
    for t in num_tags:
        num = A[:, col[t], :]
        pooled = num.sum(0) / den.sum(0)                         # (scale,)
        per = num[:, scale] / den[:, scale]
        vals.append(float(pooled[scale]))
        stat.append(float(per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else 0.0)
        scl.append(float(0.5 * (pooled.max() - pooled.min())))
    return vals, stat, scl


def _rate(process, w_tag, scale, runs):
    col, A = _runs(process)
    A = A[np.asarray(runs, int)] if runs is not None else A
    w, n = A[:, col[w_tag], :], A[:, col["norm_born"], :]
    pooled = w.sum(0) / n.sum(0); per = w[:, scale] / n[:, scale]
    stat = float(per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else 0.0
    return float(pooled[scale]), stat, float(0.5 * (pooled[1:].max() - pooled[1:].min()))


def _hyp(a, b):
    return float(np.hypot(a, b))


def load_moments(process, born=None, recoil=None, mixed=None, max_orders=None,
                 scale=None, runs=None, check_seam=True):
    r"""Turn a released table into the `(moments, config)` inputs of `upgrade()`.

    Parameters
    ----------
    process : str
        "dy", "wp", "wm", "aa", "ggh" or "zj" (or a path to a table in the same schema).
    born, recoil, mixed : list of tower keys, optional
        Override the table's default constraint set, which is the one the paper imposed
        (`table(process)["default_constraints"]`).  E.g. `recoil=["pt_w", "ctw_window"]`
        adds the W decay-angle window.
    max_orders : {tower key: N}, optional
        Cap a tower's depth.  Merged over the table's own `default_constraints.max_orders`
        (the depths the paper used where they are shallower than the table, e.g. the Z+jet
        second-jet tower at 6 of its 8 orders).
    scale : int 0..6, optional
        Pool the moments at one of the seven scale choices (`table(p)["scales"]`, 0 =
        central) instead of using the central values.  Reads <name>_runs.json.gz.
    runs : sequence of int, optional
        Pool over these run positions 0..n_runs-1 (repeats allowed: a bootstrap replica).
    check_seam : bool
        Compare each recoil window's matching scale with the power-counting rule
        alpha_s(pT) ln^2(Q/pT) < 0.2 at the table's hard scale Q, and print a warning if
        they disagree by more than 15%.  The window is compiled into the fixed-order run,
        so the check can only warn.

    Returns
    -------
    moments, config : dict, dict
        The two arguments of `upgrade(events, moments, config)`.  `events` must carry one
        array per observable named in `config` (keys = the towers' `observable` fields)
        plus the positive prior weights under 'weight', restricted to the table's
        `fiducial` selection.
    """
    t = table(process)
    dflt = t["default_constraints"]
    born = dflt["born"] if born is None else list(born)
    recoil = dflt["recoil"] if recoil is None else list(recoil)
    mixed = dflt.get("mixed", []) if mixed is None else list(mixed)
    depth = dict(dflt.get("max_orders") or {}); depth.update(max_orders or {})
    repool = scale is not None or runs is not None
    s = 0 if scale is None else int(scale)

    def tower(key):
        if key not in t["towers"]:
            raise KeyError(f"{process}: no tower '{key}'; the table has {sorted(t['towers'])}")
        d = t["towers"][key]; N = min(int(depth.get(key, d["n_orders"])), d["n_orders"])
        if repool:
            v, st, sc = _repool(process, d["numerator_tags"][:N], d["denominator_tag"], s, runs)
        else:
            v, st, sc = d["values"][:N], d["stat_errors"][:N], d["scale_errors"][:N]
        return d, v, st, sc

    def mapspec(d):
        return ("bw", d["bw"][0], d["bw"][1]) if d["map"] == "bw" else d["map"]

    moments = {"born": {}, "recoil": {}}
    config = {"born": {}, "recoil": {}}
    for key in born:
        d, v, st, sc = tower(key)
        moments["born"][d["observable"]] = dict(values=v, errors=[_hyp(a, b) for a, b in zip(st, sc)],
                                                stat_errors=st, scale_errors=sc)
        config["born"][d["observable"]] = dict(range=tuple(d["range"]), map=mapspec(d))
    for key in recoil:
        d, v, st, sc = tower(key)
        if d["kind"] != "window":
            raise ValueError(f"{process}: '{key}' is a {d['kind']} tower, not a window")
        prof = d["profile"]
        if repool:
            R, Rs, Rc = _rate(process, d["denominator_tag"], s, runs)
        else:
            R, Rs, Rc = d["window_rate"], d["window_rate_stat"], d["window_rate_scale"]
        moments["recoil"][d["observable"]] = dict(
            window_values=v, window_errors=[_hyp(a, b) for a, b in zip(st, sc)],
            stat_errors=st, scale_errors=sc, wprofile_values=v, wprofile_rate=R,
            rate=R, rate_stat=Rs, rate_scale=Rc,
            x_match=float(prof["a"]), x_hi=float(d["range"][1]), soft_lo=float(d["soft_lo"]))
        config["recoil"][d["observable"]] = dict(range=(float(d["soft_lo"]), float(d["range"][1])), map=d["map"],
                                                 soft_lo=float(d["soft_lo"]), profile=dict(prof))
        if check_seam and d.get("matching_scale_rule") and t.get("hard_scale_GeV"):
            from . import check_seam as _check
            _check(float(prof["a"]), float(t["hard_scale_GeV"]), label=f"{process} {d['observable']}")
    if mixed:
        moments["mixed"] = {}; config["mixed"] = {}
        towers_by_obs = {d["observable"]: d for d in t["towers"].values()}
        for key in mixed:
            if key not in t["joint"]:
                raise KeyError(f"{process}: no joint block '{key}'; the table has {sorted(t['joint'])}")
            j = t["joint"][key]; keys = list(j["values"])
            if repool:
                v, st, sc = _repool(process, [j["numerator_tags"][k] for k in keys], j["denominator_tag"], s, runs)
                if j["denominator_tag"] == "norm_born":
                    R, Rs, Rc = 1.0, 0.0, 0.0
                else:
                    R, Rs, Rc = _rate(process, j["denominator_tag"], s, runs)
            else:
                v = [j["values"][k] for k in keys]; st = [j["stat_errors"][k] for k in keys]
                sc = [j["scale_errors"][k] for k in keys]
                R, Rs, Rc = j["window_rate"], j["window_rate_stat"], j["window_rate_scale"]
            idx = [tuple(int(x) for x in k.split(",")) for k in keys]
            moments["mixed"][key] = dict(values=dict(zip(idx, v)), stat_errors=dict(zip(idx, st)),
                                         errors={k: _hyp(a, b) for k, a, b in zip(idx, st, sc)},
                                         rate=R, rate_stat=Rs, rate_scale=Rc)
            ox, oy = j["observables"]
            rng = {o: ((float(towers_by_obs[o]["soft_lo"]) if towers_by_obs[o]["kind"] == "window"
                        else float(towers_by_obs[o]["range"][0])), float(towers_by_obs[o]["range"][1])) for o in (ox, oy)}
            config["mixed"][key] = dict(observables=(ox, oy), range=rng,
                                        map={o: mapspec(towers_by_obs[o]) for o in (ox, oy)},
                                        profile=j["profile"], n=j["n"])
    return moments, config


def summary(process):
    """A few lines describing a table: process, runs, towers, and the paper's constraint set."""
    t = table(process); sf = t["sigma_fid_pb"]
    sig = f"sigma_fid {sf['value']:.4g} pb" if sf else "sigma_fid not validated (null)"
    lines = [f"{process}: {t['process']} at {t['order']}, {t['n_independent_runs']} runs, {sig}",
             f"   fiducial: {t['fiducial']}"]
    for k, d in t["towers"].items():
        pr = d.get("profile") or {}
        win = ("" if d["kind"] != "window" else
               f"  window rises {pr['a']:g}-{pr['b']:g}, falls {pr['c']:g}-{pr['d']:g}" if pr.get("d") is not None else
               f"  window rises {pr['a']:g}-{pr['b']:g}, cut at {pr['c']:g}")
        lines.append(f"   {d['kind']:9s} {k:12s} events['{d['observable']}']  {d['map']:3s} "
                     f"{d['range']}  {d['n_orders']} orders{win}")
    for k, d in t["joint"].items():
        lines.append(f"   joint     {k:12s} {d['observables']}  {len(d['values'])} moments")
    dc = t["default_constraints"]
    lines.append(f"   the paper's constraint set: born {dc['born']}, recoil {dc['recoil']}, mixed {dc.get('mixed', [])}"
                 + (f", depths {dc['max_orders']}" if dc.get("max_orders") else ""))
    return "\n".join(lines)
