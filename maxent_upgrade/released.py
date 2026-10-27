r"""The released fixed-order moments, as the two inputs of upgrade().

    from maxent_upgrade import upgrade, load_moments
    moments, config = load_moments("dy")          # the paper's constraint set for Drell-Yan
    result = upgrade(events, moments, config)    # events: your prior, see moments/README.md

Processes: "dy" (Z/gamma* -> l+l-, NNLO), "wp", "wm" (W+, W-, NNLO), "aa" (diphoton, NNLO),
"ggh" (gluon-fusion Higgs, NNLO, inclusive), "zj" (Z+jet, NLO).  Each table is a JSON file
in moments/ (installed as maxent_upgrade/moments/); <name>_runs.json.gz next to it holds the
channel-summed accumulators of every independent run at all seven scales, from which
`scale=` and `runs=` pool the moments again (the scale and statistical bands).  Moments are
pooled as the ratio of summed accumulators; window rates are the mean over the runs of the
per-run rate, the convention of the authors' solves.  For "ggh"
and "aa", <name>_winvar.json holds the recoil tower booked with three other matching windows
(`window="A"`, "B", "C", the matching-window variants).
"""
import gzip
import json
import os

import numpy as np

PROCESSES = {"dy": "dy_nnlo_moments", "wp": "w_nnlo_moments_WP", "wm": "w_nnlo_moments_WM",
             "aa": "aa_nnlo_moments", "ggh": "ggh_nnlo_moments", "zj": "zj_nlo_moments"}
# the matching-window variants of the recoil tower: the windows rise over about
# [2/3, 4/3] v_m (A), [4/3, 8/3] v_m (B) and [1, 4] v_m (C) instead of [1, 2] v_m
WINDOW_VARIANTS = {"ggh": "ggh_nnlo_moments_winvar", "aa": "aa_nnlo_moments_winvar"}

_RUNS_CACHE = {}


def _data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(here, "moments"), os.path.join(os.path.dirname(here), "moments")):
        if os.path.isfile(os.path.join(d, PROCESSES["dy"] + ".json")):
            return d
    raise FileNotFoundError("released moment tables not found: reinstall the package "
                            "(pip install .) or run from a clone of the repository")


def _path(name, suffix=".json"):
    key = name
    name = PROCESSES.get(name, name)
    if name.endswith(".json"):
        return name if suffix == ".json" else name[:-len(".json")] + suffix
    p = os.path.join(_data_dir(), name + suffix)
    if key not in PROCESSES and not os.path.isfile(p):
        raise ValueError(f"unknown process {key!r}: the released processes are {list(PROCESSES)} "
                         "(or pass the path of a .json table)")
    return p


def processes():
    """The process keys of the released tables."""
    return list(PROCESSES)


def table(process):
    """The raw JSON table of a process (a key of PROCESSES, or a path to a .json file)."""
    with open(_path(process)) as f:
        return json.load(f)


def window_table(process):
    """The raw JSON table of a process's matching-window variants, or None if it has none."""
    if process not in WINDOW_VARIANTS:
        return None
    with open(_path(WINDOW_VARIANTS[process])) as f:
        return json.load(f)


def windows(process):
    """{window: profile} of the matching windows released for a process's recoil tower:
    'nominal' (the central choice) and the variants 'A', 'B', 'C' for "ggh" and "aa";
    an empty dict for a process without variants."""
    wt = window_table(process)
    if wt is None:
        return {}
    out = {"nominal": dict(table(process)["towers"][wt["replaces_tower"]]["profile"])}
    out.update({V: dict(d["profile"]) for V, d in wt["windows"].items()})
    return out


def observables(process):
    """{events key: description} for every tower the table carries."""
    return {d["observable"]: d["note"] for d in table(process)["towers"].values()}


def sigma_fid(process):
    """The fixed-order cross section of a table's production inside its selection (inclusive for
    the Higgs boson), {value, stat, scale_lo, scale_hi[, note]} in pb."""
    return table(process)["sigma_fid_pb"]


def _runs(name):
    """The per-run accumulators of a table file: (tags -> column, acc[run, tag, scale], reference
    norm_born[run, scale] of the main run for a window-variant table, else None)."""
    if name not in _RUNS_CACHE:
        with gzip.open(_path(name, "_runs.json.gz"), "rt") as f:
            r = json.load(f)
        ref = r.get("reference")
        _RUNS_CACHE[name] = ({t: i for i, t in enumerate(r["tags"])}, np.asarray(r["acc"], float),
                             None if ref is None else np.asarray(ref["acc"], float))
    return _RUNS_CACHE[name]


def _sel(A, runs):
    return A[np.asarray(runs, int)] if runs is not None else A


def _repool(name, num_tags, den_tag, scale, runs):
    """Values, stat and scale errors of ratio-of-sums moments, pooled over `runs` (positions,
    repeats allowed) at `scale`, exactly as export_moments.py pools the released numbers."""
    col, A, _ = _runs(name)
    A = _sel(A, runs)
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


def _rate(name, w_tag, scale, runs):
    """Window rate <w>/sigma_fid over `runs` at `scale`, as export_moments.py and the authors'
    solves take it: the mean over the runs of the per-run rate, its run-to-run error, and the
    half-range over the seven scales of that mean.  For a window-variant table sigma_fid comes
    from the main run (its `reference` accumulators, at the same run positions): the rate is a
    ratio of the two run means, whose independent errors give its error."""
    col, A, ref = _runs(name)
    A = _sel(A, runs)
    w = A[:, col[w_tag], :]
    if ref is None:
        per = w / A[:, col["norm_born"], :]                       # (run, scale): per-run rates
        stat = float(per[:, scale].std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else 0.0
        mean = per.mean(0)
        return float(np.mean(per[:, scale])), stat, float(0.5 * (mean.max() - mean.min()))
    S = _sel(ref, runs)
    Wm, Sm = w.mean(0), S.mean(0); R = Wm / Sm
    rel = [x[:, scale].std(ddof=1) / np.sqrt(len(x)) / m[scale] if len(x) > 1 else 0.0 for x, m in ((w, Wm), (S, Sm))]
    return float(R[scale]), float(abs(R[scale]) * np.hypot(*rel)), float(0.5 * (R.max() - R.min()))


def _hyp(a, b):
    return float(np.hypot(a, b))


def load_moments(process, born=None, recoil=None, mixed=None, max_orders=None,
                 scale=None, runs=None, check_seam=True, window=None):
    r"""Turn a released table into the `(moments, config)` inputs of `upgrade()`.

    Parameters
    ----------
    process : str
        "dy", "wp", "wm", "aa", "ggh" or "zj" (or a path to a table in the same schema).
    born, recoil, mixed : list of tower keys, optional
        Override the table's default constraint set, the one the authors imposed
        (`table(process)["default_constraints"]`).  E.g. `recoil=["pt_w", "ctw_window"]`
        adds the W decay-angle window, `recoil=["ptj1", "ptj2", "pimdphi"]` the Z+jet
        acoplanarity window.
    max_orders : {tower key: N}, optional
        Cap a tower's depth.  Merged over the table's own `default_constraints.max_orders`
        (the depths the authors used where they are shallower than the table, e.g. the Z+jet
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
        so the check can only warn.  Window variants are deliberate moves of the window and
        are not checked.
    window : None, "nominal", "A", "B" or "C"
        For "ggh" and "aa": replace the recoil tower by the same tower booked with another
        matching window (`windows(process)` lists them), the inputs of the matching-
        uncertainty band.  The returned moments and config carry that window's profile and
        rate; every other tower is the main table's.

    Returns
    -------
    moments, config : dict, dict
        The two arguments of `upgrade(events, moments, config)`.  `events` must carry one
        array per observable named in `config` (keys = the towers' `observable` fields)
        plus the positive prior weights under 'weight', restricted to the table's
        `fiducial` selection.  Recoil towers use the blended transition, except where the
        authors kept the windowed form (`default_constraints.transition`, the Z+jet towers);
        a joint block with a window switches every recoil tower to 'windowed', since joint
        moments are imposed in the windowed form only.
    """
    t = table(process)
    if scale is not None and not 0 <= int(scale) < len(t["scales"]):
        raise ValueError(f"{process}: scale must be 0..{len(t['scales']) - 1} (0 = central), got {scale}")
    if runs is not None:
        runs = np.asarray(runs, int)
        n_runs = int(t["n_independent_runs"])
        if runs.ndim != 1 or runs.size == 0 or runs.min() < 0 or runs.max() >= n_runs:
            raise ValueError(f"{process}: runs must be a non-empty list of positions in 0..{n_runs - 1}")
    dflt = t["default_constraints"]
    born = dflt["born"] if born is None else list(born)
    recoil = dflt["recoil"] if recoil is None else list(recoil)
    mixed = dflt.get("mixed", []) if mixed is None else list(mixed)
    depth = dict(dflt.get("max_orders") or {}); depth.update(max_orders or {})
    repool = scale is not None or runs is not None
    s = 0 if scale is None else int(scale)
    name = PROCESSES.get(process, process)

    variant = None
    if window not in (None, "nominal"):
        wt = window_table(process)
        if wt is None:
            raise ValueError(f"{process}: no matching-window variants are released (only for {sorted(WINDOW_VARIANTS)})")
        if window not in wt["windows"]:
            raise KeyError(f"{process}: no window {window!r}; the variants are {sorted(wt['windows'])} (or 'nominal')")
        if wt["replaces_tower"] not in recoil:
            raise ValueError(f"{process}: window={window!r} replaces the recoil tower '{wt['replaces_tower']}', "
                             f"which this call does not impose (recoil={recoil})")
        variant = (wt["replaces_tower"], wt["windows"][window], WINDOW_VARIANTS[process])

    def tower(key):
        if variant is not None and key == variant[0]:
            d, src = variant[1], variant[2]
        elif key not in t["towers"]:
            raise KeyError(f"{process}: no tower '{key}'; the table has {sorted(t['towers'])}")
        else:
            d, src = t["towers"][key], name
        N = min(int(depth.get(key, d["n_orders"])), d["n_orders"])
        if repool:
            v, st, sc = _repool(src, d["numerator_tags"][:N], d["denominator_tag"], s, runs)
        else:
            v, st, sc = d["values"][:N], d["stat_errors"][:N], d["scale_errors"][:N]
        return d, src, v, st, sc

    def mapspec(d):
        return ("bw", d["bw"][0], d["bw"][1]) if d["map"] == "bw" else d["map"]

    # joint moments are imposed in the windowed form, so a joint block with a window keeps
    # every recoil tower windowed (the default Z+jet configuration); otherwise the table says
    for key in mixed:
        if key not in t["joint"]:
            raise KeyError(f"{process}: no joint block '{key}'; the table has {sorted(t['joint'])}")
    windowed_joint = any(t["joint"][k].get("profile") for k in mixed)
    transition = dict(dflt.get("transition") or {})

    moments = {"born": {}, "recoil": {}}
    config = {"born": {}, "recoil": {}}
    for key in born:
        d, _, v, st, sc = tower(key)
        moments["born"][d["observable"]] = dict(values=v, errors=[_hyp(a, b) for a, b in zip(st, sc)],
                                                stat_errors=st, scale_errors=sc)
        config["born"][d["observable"]] = dict(range=tuple(d["range"]), map=mapspec(d))
    for key in recoil:
        d, src, v, st, sc = tower(key)
        if d["kind"] != "window":
            raise ValueError(f"{process}: '{key}' is a {d['kind']} tower, not a window")
        prof = d["profile"]
        if repool:
            R, Rs, Rc = _rate(src, d["denominator_tag"], s, runs)
        else:
            R, Rs, Rc = d["window_rate"], d["window_rate_stat"], d["window_rate_scale"]
        moments["recoil"][d["observable"]] = dict(
            window_values=v, window_errors=[_hyp(a, b) for a, b in zip(st, sc)],
            stat_errors=st, scale_errors=sc, wprofile_values=v, wprofile_rate=R,
            rate=R, rate_stat=Rs, rate_scale=Rc,
            x_match=float(prof["a"]), x_hi=float(d["range"][1]), soft_lo=float(d["soft_lo"]))
        config["recoil"][d["observable"]] = dict(range=(float(d["soft_lo"]), float(d["range"][1])), map=d["map"],
                                                 soft_lo=float(d["soft_lo"]), profile=dict(prof))
        tr = "windowed" if windowed_joint else transition.get(key)
        if tr:
            config["recoil"][d["observable"]]["transition"] = tr
        if (check_seam and src == name and d.get("matching_scale_rule") and t.get("hard_scale_GeV")):
            from . import check_seam as _check
            _check(float(prof["a"]), float(t["hard_scale_GeV"]), label=f"{process} {d['observable']}")
    if mixed:
        moments["mixed"] = {}; config["mixed"] = {}
        towers_by_obs = {d["observable"]: d for d in t["towers"].values()}
        for key in mixed:
            j = t["joint"][key]; keys = list(j["values"])
            if repool:
                v, st, sc = _repool(name, [j["numerator_tags"][k] for k in keys], j["denominator_tag"], s, runs)
                if j["denominator_tag"] == "norm_born":
                    R, Rs, Rc = 1.0, 0.0, 0.0
                else:
                    R, Rs, Rc = _rate(name, j["denominator_tag"], s, runs)
            else:
                v = [j["values"][k] for k in keys]; st = [j["stat_errors"][k] for k in keys]
                sc = [j["scale_errors"][k] for k in keys]
                R, Rs, Rc = j["window_rate"], j["window_rate_stat"], j["window_rate_scale"]
            idx = [tuple(int(x) for x in k.split(",")) for k in keys]
            moments["mixed"][key] = dict(values=dict(zip(idx, v)), stat_errors=dict(zip(idx, st)),
                                         scale_errors=dict(zip(idx, sc)),
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
    """A few lines describing a table: process, provenance, towers, and the default constraint set."""
    t = table(process); sf = t["sigma_fid_pb"]; pv = t.get("provenance", {})
    lines = [f"{process}: {t['process']} at {t['order']}, {t['n_independent_runs']} runs, "
             f"sigma {sf['value']:.4g} pb (stat {sf['stat']:.2g}, scale {sf['scale_lo']:.4g}-{sf['scale_hi']:.4g})",
             f"   fiducial: {t['fiducial']}"]
    if sf.get("note"):
        lines.append(f"   sigma: {sf['note']}")
    if pv:
        lines.append(f"   NNLOJET run {pv['run_directory']} ({pv['run_name']}), run card {pv['runcard']}"
                     + (f", decay_type = {pv['decay_type']}" if "decay_type" in pv else ""))
    for k, d in t["towers"].items():
        pr = d.get("profile") or {}
        win = ("" if d["kind"] != "window" else
               f"  window rises {pr['a']:g}-{pr['b']:g}, falls {pr['c']:g}-{pr['d']:g}" if pr.get("d") is not None else
               f"  window rises {pr['a']:g}-{pr['b']:g}, cut at {pr['c']:g}")
        lines.append(f"   {d['kind']:9s} {k:12s} events['{d['observable']}']  {d['map']:3s} "
                     f"{d['range']}  {d['n_orders']} orders{win}")
    for k, d in t["joint"].items():
        lines.append(f"   joint     {k:12s} {d['observables']}  {len(d['values'])} moments")
    wt = window_table(process)
    if wt is not None:
        lines.append(f"   window variants of {wt['replaces_tower']} (load_moments(..., window=V)): " + ", ".join(
            f"{V} rises {d['profile']['a']:g}-{d['profile']['b']:g}" for V, d in wt["windows"].items()))
    dc = t["default_constraints"]
    lines.append(f"   default constraint set: born {dc['born']}, recoil {dc['recoil']}, mixed {dc.get('mixed', [])}"
                 + (f", depths {dc['max_orders']}" if dc.get("max_orders") else "")
                 + (f", transition {dc['transition']}" if dc.get("transition") else ", blended recoil towers"))
    return "\n".join(lines)
