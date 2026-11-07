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
per-run rate, the convention of the authors' solves.  <name>_winvar.json holds the recoil
tower (for "zj" both jet towers and their joint block) booked with other matching windows
(`window="A"`, "B", "C", and "D" for "ggh": the matching-window variants).  For "wp", "wm"
and "ggh", <name>_pdf.json holds the moments of the CT18NNLO members (`pdf_member=k`).
"""
import gzip
import json
import os
import re

import numpy as np

PROCESSES = {"dy": "dy_nnlo_moments", "wp": "w_nnlo_moments_WP", "wm": "w_nnlo_moments_WM",
             "aa": "aa_nnlo_moments", "ggh": "ggh_nnlo_moments", "zj": "zj_nlo_moments"}
# the matching-window variants of the recoil tower: the windows rise over about
# [2/3, 4/3] v_m (A), [4/3, 8/3] v_m (B) and [1, 4] v_m (C) instead of [1, 2] v_m;
# for the Higgs boson D rises over [54, 108] GeV, the color-corrected matching scale
WINDOW_VARIANTS = {p: PROCESSES[p] + "_winvar" for p in PROCESSES}
# the moments of the CT18NNLO members 0..58 (the PDF band)
PDF_MEMBERS = {p: PROCESSES[p] + "_pdf" for p in ("wp", "wm", "ggh")}
# The NNLO Z+jet calculation of the letter (STRIPPER, R. Poncelet), as produced: its profiled
# pT_ll moments (block pT_Z_cm_rm1_w1, the 30-60 GeV window cut at 500 GeV, log map on
# [30, 2500] GeV) replace the Drell-Yan recoil tower with load_moments("dy", nnlo_zj=True).
NNLO_ZJ_FILE = "ppzj-moments_NNLO_m0.xml"
# NNLOJET scale choice -> STRIPPER's, which orders (1,1),(2,2),(.5,.5),(1,.5),(1,2),(.5,1),(2,1)
_TO_STRIPPER = [0, 1, 2, 6, 4, 5, 3]

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


def _variant(wt, V):
    """({tower key: tower}, {joint key: block}) that window V of a window-variant table replaces."""
    W = wt["windows"][V]
    return (W["towers"] if "towers" in W else {wt["replaces_tower"]: W}), W.get("joint", {})


def windows(process):
    """{window: profile} of the matching windows released for a process's recoil tower:
    'nominal' (the central choice) and the variants 'A', 'B', 'C' (and 'D' for "ggh");
    an empty dict for a process without variants.  For "zj" the profile is that of both jets."""
    wt = window_table(process)
    if wt is None:
        return {}
    key = (wt.get("replaces_towers") or [wt["replaces_tower"]])[0]
    out = {"nominal": dict(table(process)["towers"][key]["profile"])}
    out.update({V: dict(d["profile"]) for V, d in wt["windows"].items()})
    return out


def pdf_table(process):
    """The raw JSON table of a process's CT18NNLO member moments, or None if it has none."""
    if process not in PDF_MEMBERS:
        return None
    with open(_path(PDF_MEMBERS[process])) as f:
        return json.load(f)


def hessian_band(x, cl68=True):
    """The CT18 Hessian band of a quantity x[k] computed with each member k = 0..58 (e.g. a
    histogram per member, x of shape (59, n_bins)): the asymmetric (plus, minus) deviations from
    member 0, from the eigenvector pairs (2i-1, 2i).  CT18 is a 90% CL set: cl68=True divides
    by 1.645 for a 68% CL band."""
    x = np.asarray(x, float)
    x0 = x[0]
    up, dn = np.zeros_like(x0), np.zeros_like(x0)
    for i in range(1, len(x) - 1, 2):
        a, b = x[i] - x0, x[i + 1] - x0
        up += np.maximum(np.maximum(a, b), 0.0) ** 2
        dn += np.maximum(np.maximum(-a, -b), 0.0) ** 2
    f = 1.0 / 1.645 if cl68 else 1.0
    return f * np.sqrt(up), f * np.sqrt(dn)


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


def _rate(name, w_tag, scale, runs, norm_tag="norm_born"):
    """Window rate <w>/sigma_fid over `runs` at `scale`, as export_moments.py and the authors'
    solves take it: the mean over the runs of the per-run rate, its run-to-run error, and the
    half-range over the seven scales of that mean.  For a window-variant table sigma_fid comes
    from the main run (its `reference` accumulators, at the same run positions): the rate is a
    ratio of the two run means, whose independent errors give its error.  A tower integrated
    from a histogram books its per-run rate directly (`norm_tag` = 1)."""
    col, A, ref = _runs(name)
    A = _sel(A, runs)
    w = A[:, col[w_tag], :]
    if ref is None:
        per = w / A[:, col[norm_tag], :]                          # (run, scale): per-run rates
        stat = float(per[:, scale].std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else 0.0
        mean = per.mean(0)
        return float(np.mean(per[:, scale])), stat, float(0.5 * (mean.max() - mean.min()))
    S = _sel(ref, runs)
    Wm, Sm = w.mean(0), S.mean(0); R = Wm / Sm
    rel = [x[:, scale].std(ddof=1) / np.sqrt(len(x)) / m[scale] if len(x) > 1 else 0.0 for x, m in ((w, Wm), (S, Sm))]
    return float(R[scale]), float(abs(R[scale]) * np.hypot(*rel)), float(0.5 * (R.max() - R.min()))


def _stripper_block(path, block):
    """One block of a STRIPPER XML file: values (bin, 7 scales) and central errors (bin),
    under- and overflow slots removed."""
    s = open(path).read()
    for o in re.findall(r"<observable>(.*?)</observable>", s, re.S):
        if re.search(r"<description>(.*?)</description>", o, re.S).group(1).strip() == block:
            bins = re.findall(r"<bin>(.*?)</bin>", o, re.S)
            V = np.array([[float(x) for x in re.search(r"<values>(.*?)</values>", b, re.S).group(1).split(",")] for b in bins])
            E = np.array([[float(x) for x in re.search(r"<errors>(.*?)</errors>", b, re.S).group(1).split(",")] for b in bins])
            return V[1:-1, :7], E[1:-1, 0]
    raise KeyError(f"{path}: no block {block!r}")


def _nnlo_zj_recoil(scale, runs):
    """The paper's NNLO Z+jet pT_ll tower at an NNLOJET scale choice, its window rate taken
    against the Drell-Yan fiducial cross section of the same scale choice and runs."""
    W, e = _stripper_block(os.path.join(_data_dir(), NNLO_ZJ_FILE), "pT_Z_cm_rm1_w1")
    col, A, _ = _runs(PROCESSES["dy"])
    sig = 2.002e-3 * float(_sel(A, runs)[:, col["norm_born"], scale].mean())   # pb
    j = _TO_STRIPPER[scale]; W0 = W[0, j]
    v = W[1:, j] / W0
    st = np.sqrt((e[1:] / W0) ** 2 + (W[1:, 0] * e[0] / W0 ** 2) ** 2)
    r = W[1:, :] / W[0, :]; sc = 0.5 * (r.max(1) - r.min(1))
    R = float(W0 / sig)
    v, st, sc = [float(x) for x in v], [float(x) for x in st], [float(x) for x in sc]
    return dict(window_values=v, window_errors=[_hyp(a, b) for a, b in zip(st, sc)],
                stat_errors=st, scale_errors=sc, wprofile_values=v, wprofile_rate=R,
                rate=R, rate_stat=float(e[0] / sig), rate_scale=float(0.5 * (W[0].max() - W[0].min()) / sig),
                x_match=30.0, x_hi=2500.0, soft_lo=30.0)


def _hyp(a, b):
    return float(np.hypot(a, b))


def load_moments(process, born=None, recoil=None, mixed=None, max_orders=None,
                 scale=None, runs=None, check_seam=True, window=None, nnlo_zj=False, pdf_member=None,
                 qt_basis="range"):
    r"""Turn a released table into the `(moments, config)` inputs of `upgrade()`.

    Parameters
    ----------
    process : str
        "dy", "wp", "wm", "aa", "ggh" or "zj" (or a path to a table in the same schema).
    born, recoil, mixed : list of tower keys, optional
        Override the table's default constraint set, the one the authors imposed
        (`table(process)["default_constraints"]`).  E.g. `recoil=["pt_w", "ctw_window"]`
        adds the W decay-angle window, `recoil=["ptj1", "ptj2", "aco"]` the Z+jet
        acoplanarity moments of the paper (window 0.687 -> 1.374), and
        `recoil=["ptj1", "ptj2", "pimdphi"]` the earlier acoplanarity window (0.05 -> 0.2).
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
    window : None, "nominal", "A", "B", "C" or (for "ggh") "D"
        Replace the recoil tower (for "zj" both jet towers and their joint block) by the same
        tower booked with another matching window (`windows(process)` lists them), the
        inputs of the matching-uncertainty band.  The returned moments and config carry that
        window's profile and rate; every other tower is the main table's.  The variants of
        "dy", "wp", "wm", "zj" and the Higgs D are central-scale only (no `scale`, `runs`).
    nnlo_zj : bool
        For "dy": replace the p_T^ll tower by the letter's NNLO Z+jet moments (STRIPPER,
        R. Poncelet; moments/ppzj-moments_NNLO_m0.xml), as in its MaxEnt NNLO(Zj) solve.
        They carry Monte Carlo errors but no per-run values, so `runs` resamples only the
        Drell-Yan towers and the cross section their rate is taken against.
    pdf_member : int 0..58, optional
        For "wp", "wm" and "ggh": add to the central moments the difference between CT18NNLO
        member k and member 0 (`pdf_table(process)`), the inputs of the PDF band.
        `hessian_band` forms the band from the 59 re-solves.
    qt_basis : "range" (default) or "booked"
        The polynomials of each logarithmically mapped window tower (the q_T and jet
        towers).  Their measurement functions vary only between the window's lower edge,
        below which they freeze (blended) or vanish (windowed), and its cut.  NNLOJET booked
        them as Chebyshev polynomials of the log map on a wider interval, on which they are
        nearly linearly dependent.  "range" re-expresses them as Chebyshev polynomials of the
        log map on the interval where they act (`acting_range`).  The two maps are affine in
        ln v, so the new moments are exact combinations of the booked ones: values, run-to-run
        errors and scale envelopes are pooled again from the per-run accumulators.  A variant
        released without per-run accumulators takes the errors of the main table's tower in
        the same basis.  The posterior is unchanged in exact arithmetic, but the solver's ridge
        no longer touches directions the input resolves, and the signal-to-noise selection
        judges each degree of the acting range.  "booked" returns the moments as booked (the
        letter and the earlier releases).  The letter's NNLO Z+jet tower (`nnlo_zj=True`)
        carries uncorrelated errors per degree only and stays as booked.

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

    variant, vjoint = {}, {}
    if window not in (None, "nominal"):
        wt = window_table(process)
        if wt is None:
            raise ValueError(f"{process}: no matching-window variants are released (only for {sorted(WINDOW_VARIANTS)})")
        if window not in wt["windows"]:
            raise KeyError(f"{process}: no window {window!r}; the variants are {sorted(wt['windows'])} (or 'nominal')")
        vt, vjoint = _variant(wt, window)
        if not any(k in recoil for k in vt):
            raise ValueError(f"{process}: window={window!r} replaces the recoil tower(s) {list(vt)}, "
                             f"which this call does not impose (recoil={recoil})")
        if repool and any("numerator_tags" not in d for d in vt.values()):
            raise ValueError(f"{process}: window {window!r} is released at the central scale only; "
                             f"it does not combine with scale= or runs=")
        variant = {k: (d, WINDOW_VARIANTS[process]) for k, d in vt.items()}
    if pdf_member is not None:
        pt = pdf_table(process)
        if pt is None:
            raise ValueError(f"{process}: no CT18NNLO member moments are released (only for {sorted(PDF_MEMBERS)})")
        if not 0 <= int(pdf_member) < int(pt["n_members"]):
            raise ValueError(f"{process}: pdf_member must be 0..{pt['n_members'] - 1}, got {pdf_member}")
        if repool or variant or nnlo_zj:
            raise ValueError(f"{process}: pdf_member shifts the central moments; it does not combine with "
                             f"scale=, runs=, window= or nnlo_zj=")

    def tower(key):
        if key in variant:
            d, src = variant[key]
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

    if qt_basis not in ("range", "booked"):
        raise ValueError(f"qt_basis must be 'range' or 'booked', got {qt_basis!r}")
    moments = {"born": {}, "recoil": {}}
    config = {"born": {}, "recoil": {}}
    booked = {}
    for key in born:
        d, _, v, st, sc = tower(key)
        moments["born"][d["observable"]] = dict(values=v, errors=[_hyp(a, b) for a, b in zip(st, sc)],
                                                stat_errors=st, scale_errors=sc)
        config["born"][d["observable"]] = dict(range=tuple(d["range"]), map=mapspec(d))
    for key in recoil:
        d, src, v, st, sc = tower(key)
        if d["kind"] != "window":
            raise ValueError(f"{process}: '{key}' is a {d['kind']} tower, not a window")
        if d["observable"] in moments["recoil"]:
            raise ValueError(f"{process}: two recoil towers in {recoil} constrain events['{d['observable']}']; "
                             f"impose one of them")
        prof = d["profile"]
        booked[d["observable"]] = dict(d=d, src=src, N=len(v), key=key)
        if repool:
            R, Rs, Rc = _rate(src, d["denominator_tag"], s, runs, d.get("rate_norm_tag", "norm_born"))
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
            j = dict(t["joint"][key], **vjoint.get(key, {})); keys = list(j["values"])
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
    if nnlo_zj:
        if name != PROCESSES["dy"] or "pT_ll" not in moments["recoil"]:
            raise ValueError("nnlo_zj=True replaces the Drell-Yan p_T^ll tower: use it with load_moments(\"dy\")")
        moments["recoil"]["pT_ll"] = _nnlo_zj_recoil(s, runs)
    if pdf_member is not None:
        k = int(pdf_member)
        for key, d in pt["towers"].items():
            dv = np.asarray(d["values"][k], float) - np.asarray(d["values"][0], float)
            if key in born:
                b = moments["born"][d["observable"]]
                b["values"] = list(np.asarray(b["values"]) + dv[:len(b["values"])])
            elif key in recoil:
                rc = moments["recoil"][d["observable"]]
                rc["window_values"] = list(np.asarray(rc["window_values"]) + dv[:len(rc["window_values"])])
                rc["wprofile_values"] = rc["window_values"]
                rc["rate"] = rc["wprofile_rate"] = float(rc["rate"] + (d["window_rate"][k] - d["window_rate"][0]))
    if qt_basis == "range":
        for obs, info in booked.items():
            if nnlo_zj and obs == "pT_ll":
                continue
            _to_acting_range(moments["recoil"][obs], config["recoil"][obs], info, t, name, s, runs)
    return moments, config


def acting_range(tower, profile=None):
    """The interval where a window tower's measurement functions vary: from the larger of the
    booked lower end (soft_lo) and the window's lower edge a, below which the blended functions
    freeze and the windowed ones vanish, up to the window's cut c (or the end d of a smooth top),
    within the booked interval."""
    prof = tower["profile"] if profile is None else profile
    lo, hi = float(tower["soft_lo"]), float(tower["range"][1])
    a, c, e = float(prof["a"]), prof.get("c"), prof.get("d")
    top = float(e) if (e is not None and c is not None and float(e) > float(c)) else (float(c) if c is not None else hi)
    return max(lo, a), min(top, hi)


def cheb_affine(N, alpha, beta):
    """A[n, k] with T_n(alpha x + beta) = sum_k A[n, k] T_k(x), n, k = 0..N (exact for polynomials)."""
    C = np.polynomial.chebyshev
    lin = C.Chebyshev([beta, alpha])
    A = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        cf = C.Chebyshev.basis(n)(lin).coef
        A[n, :len(cf)] = cf
    return A


def _log_u(v, lo, hi):
    return 2.0 * (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) - 1.0


def _range_accumulators(src, d, N, A, scale, runs):
    """Values, run-to-run and scale errors of a window tower's first N moments in the basis
    T_n(alpha x + beta), pooled from the per-run accumulators like `_repool`."""
    col, Acc, _ = _runs(src)
    Acc = _sel(Acc, runs)
    den = Acc[:, col[d["denominator_tag"]], :]
    num = np.stack([Acc[:, col[tg], :] for tg in d["numerator_tags"][:N]])
    new = A[1:N + 1, 0][:, None, None] * den[None] + np.einsum("nk,krs->nrs", A[1:N + 1, 1:N + 1], num)
    pooled = new.sum(1) / den.sum(0)[None]
    per = new[:, :, scale] / den[None, :, scale]
    st = per.std(1, ddof=1) / np.sqrt(per.shape[1]) if per.shape[1] > 1 else np.zeros(N)
    return pooled[:, scale], st, 0.5 * (pooled.max(1) - pooled.min(1))


def _variant_errors(name, main, d, N, A):
    """Run-to-run and scale errors of a variant tower in the basis A.  A variant released without per-run values
    carries only its booked errors per degree, so the main tower's errors in the same basis (from its runs) are scaled
    by the median ratio of the variant's booked errors to the main tower's (the variant runs are fewer).  The estimate
    serves a selection only: re-solves for the transition-region band hold the central solve's degrees."""
    _, st, sc = _range_accumulators(name, main, N, A, 0, None)

    def ratio(key):
        v, m = np.asarray(d[key][:N], float), np.asarray(main[key][:N], float)
        ok = np.isfinite(v) & np.isfinite(m) & (m > 0) & (v > 0)
        return float(np.median(v[ok] / m[ok])) if ok.any() else 1.0
    return st * ratio("stat_errors"), sc * ratio("scale_errors")


def _to_acting_range(rc, cfg, info, t, name, scale, runs):
    """Re-express one log-mapped window tower (moments `rc`, config `cfg`) as Chebyshev
    polynomials of the log map on its acting range, in place."""
    d, src, N = info["d"], info["src"], len(rc["window_values"])
    if d.get("map") != "log" or d.get("kind") != "window":
        return
    lo, hi = float(d["soft_lo"]), float(d["range"][1])
    L, U = acting_range(d)
    main = t["towers"][info["key"]]                 # the main table's tower that this one is or replaces
    if src != name:
        # a window variant: the interval covering its own acting range and the main tower's, so that the variant
        # imposes the central solve's polynomials wherever its range allows and differs from it only by its window
        Lm, Um = acting_range(main)
        L, U = min(L, Lm), max(U, Um)
    if abs(L - lo) <= 1e-12 * lo and abs(U - hi) <= 1e-12 * hi:
        return
    xm, xc = _log_u(L, lo, hi), _log_u(U, lo, hi)
    al = 2.0 / (xc - xm)
    A = cheb_affine(N, al, -1.0 - al * xm)
    v = A[1:, 0] + A[1:, 1:] @ np.asarray(rc["window_values"], float)
    if "numerator_tags" in d and os.path.exists(_path(src, "_runs.json.gz")):
        _, st, sc = _range_accumulators(src, d, N, A, scale, runs)
    else:
        # a variant released without per-run values: the main tower's errors, scaled (_variant_errors)
        if (float(main["soft_lo"]), float(main["range"][1])) != (lo, hi):
            raise ValueError(f"{name}: variant of {d['observable']} booked on another interval than the main tower")
        st, sc = _variant_errors(name, main, d, N, A)
    st, sc = [float(x) for x in st], [float(x) for x in sc]
    rc.update(window_values=[float(x) for x in v], wprofile_values=[float(x) for x in v],
              stat_errors=st, scale_errors=sc, window_errors=[_hyp(a_, b_) for a_, b_ in zip(st, sc)],
              soft_lo=L, x_hi=U)
    cfg.update(range=(L, U), soft_lo=L)


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
        rep = wt.get("replaces_towers") or [wt["replaces_tower"]]
        lines.append(f"   window variants of {'+'.join(rep)} (load_moments(..., window=V)): " + ", ".join(
            f"{V} rises {d['profile']['a']:g}-{d['profile']['b']:g}" for V, d in wt["windows"].items()))
    pt = pdf_table(process)
    if pt is not None:
        lines.append(f"   CT18NNLO members 0-{pt['n_members'] - 1} of {list(pt['towers'])} "
                     f"(load_moments(..., pdf_member=k))")
    dc = t["default_constraints"]
    lines.append(f"   default constraint set: born {dc['born']}, recoil {dc['recoil']}, mixed {dc.get('mixed', [])}"
                 + (f", depths {dc['max_orders']}" if dc.get("max_orders") else "")
                 + (f", transition {dc['transition']}" if dc.get("transition") else ", blended recoil towers"))
    return "\n".join(lines)
