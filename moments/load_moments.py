#!/usr/bin/env python3
r"""Turn a released moment table (moments/<process>.json) into the two inputs of
maxent_upgrade.upgrade(): the `moments` dict and the matching `config`.

    from moments.load_moments import load
    moments, config = load("dy")                     # the constraint set of the paper
    result = upgrade(events, moments, config)

`events` is your prior, a dict of equal-length arrays with the observables the
table names (`observables()` lists them), defined exactly as in the table's
"fiducial" string and in MOMENT_CONSTRUCTION.md, plus the positive weights
under 'weight'.  Both sides must integrate the same phase space.

Options: `born`, `recoil`, `mixed` override the default constraint set with any
towers the table carries (e.g. load("wp", recoil=["pt_w", "ctw_window"]) adds
the W decay-angle window); `max_orders` caps a tower's depth ({tower: N}).
The solver's resolution criterion (moment_selection=True) still decides how
many of the supplied orders are imposed.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"dy": "dy_nnlo_moments", "wp": "w_nnlo_moments_WP", "wm": "w_nnlo_moments_WM",
         "aa": "aa_nnlo_moments", "ggh": "ggh_nnlo_moments", "zj": "zj_nlo_moments"}


def table(process):
    """The raw JSON table of a process ('dy', 'wp', 'wm', 'aa', 'ggh', 'zj')."""
    name = FILES.get(process, process)
    path = name if name.endswith(".json") else os.path.join(HERE, name + ".json")
    with open(path) as f:
        return json.load(f)


def observables(process):
    """{events key: description} for every tower the table carries."""
    t = table(process)
    out = {d["observable"]: d["note"] for d in t["towers"].values()}
    return out


def _hyp(a, b):
    return (a * a + b * b) ** 0.5


def load(process, born=None, recoil=None, mixed=None, max_orders=None):
    t = table(process); dflt = t["default_constraints"]
    born = dflt["born"] if born is None else born
    recoil = dflt["recoil"] if recoil is None else recoil
    mixed = dflt["mixed"] if mixed is None else mixed
    max_orders = max_orders or {}
    moments = {"born": {}, "recoil": {}}
    config = {"born": {}, "recoil": {}}
    for key in born:
        d = t["towers"][key]; N = max_orders.get(key, d["n_orders"])
        obs = d["observable"]
        moments["born"][obs] = dict(values=d["values"][:N],
                                    errors=[_hyp(s, c) for s, c in zip(d["stat_errors"][:N], d["scale_errors"][:N])],
                                    stat_errors=d["stat_errors"][:N], scale_errors=d["scale_errors"][:N])
        mp = ("bw", d["bw"][0], d["bw"][1]) if d["map"] == "bw" else d["map"]
        config["born"][obs] = dict(range=tuple(d["range"]), map=mp)
    for key in recoil:
        d = t["towers"][key]; N = max_orders.get(key, d["n_orders"])
        if d["kind"] != "window":
            raise ValueError(f"{key} is a {d['kind']} tower, not a windowed recoil")
        obs = d["observable"]; prof = d["profile"]
        errs = [_hyp(s, c) for s, c in zip(d["stat_errors"][:N], d["scale_errors"][:N])]
        moments["recoil"][obs] = dict(window_values=d["values"][:N], window_errors=errs,
                                      wprofile_values=d["values"][:N], wprofile_rate=d["window_rate"],
                                      rate=d["window_rate"], rate_stat=d["window_rate_stat"], rate_scale=d["window_rate_scale"],
                                      x_match=float(prof["a"]), x_hi=float(d["range"][1]), soft_lo=float(d["soft_lo"]))
        config["recoil"][obs] = dict(range=(float(d["soft_lo"]), float(d["range"][1])), map=d["map"],
                                     soft_lo=float(d["soft_lo"]), profile=dict(prof))
    if mixed:
        moments["mixed"] = {}; config["mixed"] = {}
        for key in mixed:
            j = t["joint"][key]
            vals = {tuple(int(x) for x in k.split(",")): v for k, v in j["values"].items()}
            stat = {tuple(int(x) for x in k.split(",")): v for k, v in j["stat_errors"].items()}
            errs = {k: _hyp(stat[k], j["scale_errors"][f"{k[0]},{k[1]}"]) for k in vals}
            moments["mixed"][key] = dict(values=vals, stat_errors=stat, errors=errs, rate=j["window_rate"],
                                         rate_stat=j["window_rate_stat"], rate_scale=j["window_rate_scale"])
            ox, oy = j["observables"]
            tw = {d["observable"]: d for d in t["towers"].values()}
            rng = {o: (float(tw[o]["soft_lo"]) if tw[o]["kind"] == "window" else float(tw[o]["range"][0]), float(tw[o]["range"][1])) for o in (ox, oy)}
            mp = {o: tw[o]["map"] for o in (ox, oy)}
            config["mixed"][key] = dict(observables=(ox, oy), range=rng, map=mp, profile=j["profile"], n=j["n"])
    return moments, config


if __name__ == "__main__":
    import sys
    for p in (sys.argv[1:] or list(FILES)):
        t = table(p)
        print(f"{p}: {t['process']} at {t['order']}, {t['n_independent_runs']} runs, sigma_fid {t['sigma_fid_pb']['value']:.3g} pb")
        for k, d in t["towers"].items():
            print(f"   {d['kind']:9s} {k:12s} events['{d['observable']}']  {d['map']:3s} {d['range']}  {d['n_orders']} orders")
        for k, d in t["joint"].items():
            print(f"   joint     {k:12s} {d['observables']}  {len(d['values'])} moments")
        print(f"   paper's constraint set: {t['default_constraints']}")
