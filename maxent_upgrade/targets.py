"""Portable Chebyshev targets identified by observable and polynomial degree."""
from copy import deepcopy

import numpy as np

from .constraints import compute_fo_moments
from .uncertainty import Variation, upgrade_ensemble


def _number(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int, np.number)):
        raise ValueError(f"{name} must be a number")
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} is out of range: {value}")
    return float(value)


def _fields(record, allowed, label):
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a JSON object")
    extra = set(record).difference(allowed)
    if extra:
        raise ValueError(f"unrecognized {label} fields: {sorted(extra)}")


def _mapping(name, interval, mapping):
    values = np.asarray(interval, float)
    if values.shape != (2,) or not np.isfinite(values).all() or values[1] <= values[0]:
        raise ValueError(f"{name}: range must contain two finite, increasing endpoints")
    if isinstance(mapping, (tuple, list)) and len(mapping) == 3 and mapping[0] == "bw":
        if all(_number(v, name) > 0 for v in mapping[1:]):
            return
    elif mapping == "lin" or (mapping == "log" and values[0] > 0):
        return
    raise ValueError(f"{name}: use lin, log with positive endpoints, or [bw, pole mass, width]")


def _profile(name, profile):
    _fields(profile, ("a", "b", "c", "d"), f"{name} window")
    a, b = _number(profile["a"], "a"), _number(profile["b"], "b")
    c, d = profile.get("c"), profile.get("d")
    if not 0 < a <= b or (c is not None and _number(c, "c") < b):
        raise ValueError(f"{name}: window edges must satisfy 0 < a <= b <= c")
    if d is not None and (c is None or _number(d, "d") <= c):
        raise ValueError(f"{name}: a smooth upper window requires d > c")


def _joint_maps(config):
    """Each named axis has one map. Joint blocks may refer to an existing axis by ID."""
    if not isinstance(config, dict):
        raise ValueError("config must be an object of observable definitions")
    config = deepcopy(config)
    for role in ("born", "recoil", "mixed"):
        blocks = config.get(role, {})
        if not isinstance(blocks, dict) or not all(isinstance(spec, dict) for spec in blocks.values()):
            raise ValueError(f"{role} must map observable IDs to definition objects")
    axes = {}
    for role in ("born", "recoil"):
        for name, spec in config.get(role, {}).items():
            _mapping(name, spec["range"], spec["map"])
            interval = list(spec["range"])
            if role == "recoil" and "soft_lo" in spec:
                interval[0] = spec["soft_lo"]
            axes[name] = {"range": interval, "map": spec["map"]}
    for block in config.get("mixed", {}).values():
        for field in ("range", "map"):
            for axis, value in block.get(field, {}).items():
                defined = axes.setdefault(axis, {})
                if field in defined and not np.array_equal(value, defined[field]):
                    raise ValueError(f"conflicting {field} definitions for observable ID {axis!r}")
                defined[field] = value
    for block in config.get("mixed", {}).values():
        for axis in block["observables"]:
            for field in ("range", "map"):
                if field not in axes.get(axis, {}):
                    raise ValueError(f"declare the {field} of joint observable ID {axis!r}")
                block.setdefault(field, {})[axis] = axes[axis][field]
    return config


def _definitions(config):
    _fields(config, ("born", "recoil", "mixed", "weight_key", "moment_selection", "fixed_orders",
                     "snr_threshold", "L2", "soft_constraints", "soft_errors"), "config")
    for flag in ("moment_selection", "soft_constraints"):
        if flag in config and type(config[flag]) is not bool:
            raise ValueError(f"{flag} must be true or false")
    for key in ("snr_threshold", "L2"):
        if key in config:
            _number(config[key], key, minimum=0)
    if config.get("soft_errors", "stat") not in ("stat", "total"):
        raise ValueError("soft_errors must be stat or total")
    roles = {}
    for role in ("born", "recoil", "mixed"):
        for name, spec in config.get(role, {}).items():
            if not isinstance(name, str) or not name or name in roles:
                raise ValueError(f"observable IDs must be unique across all constraints: {name!r}")
            roles[name] = role
            if role == "mixed":
                _fields(spec, ("observables", "range", "map", "n", "profile", "transition", "include_rate"), name)
                axes = spec["observables"]
                if (len(axes) != 2 or axes[0] == axes[1] or type(spec["n"]) is not int
                        or spec["n"] < 1):
                    raise ValueError(f"{name}: give two distinct axes in order and a positive joint degree n")
                profiles = spec.get("profile") or {}
                if set(spec["range"]) != set(axes) or set(spec["map"]) != set(axes) or set(profiles) - set(axes):
                    raise ValueError(f"{name}: joint map, range and window IDs must match the named axes")
                for axis, profile in profiles.items():
                    if profile:
                        _profile(axis, profile)
                for axis in axes:
                    _mapping(axis, spec["range"][axis], spec["map"][axis])
            else:
                allowed = ("range", "map", "profile", "transition", "soft_lo") if role == "recoil" else ("range", "map")
                _fields(spec, allowed, name)
                if spec.get("profile"):
                    _profile(name, spec["profile"])
                if "soft_lo" in spec and not 0 < _number(spec["soft_lo"], "soft_lo") < spec["range"][1]:
                    raise ValueError(f"{name}: soft_lo must be positive and below the upper map endpoint")
    if not roles:
        raise ValueError("config must define at least one observable in born, recoil or mixed")
    for name, order in config.get("fixed_orders", {}).items():
        if roles.get(name) not in ("born", "recoil") or type(order) is not int or order < int(roles[name] == "recoil"):
            raise ValueError(f"invalid fixed tower degree {name}: {order}")
    return roles


def _read_values(entry, *, need_error=True):
    value = _number(entry["value"], "value")
    if "error" not in entry:
        if need_error or "stat_error" in entry:
            raise ValueError("supply error for automatic degree selection, or set moment_selection to false "
                             "to impose the supplied degrees without input errors")
        return value, None, None
    error = _number(entry["error"], "error", minimum=0)
    stat = _number(entry.get("stat_error", error), "stat_error", minimum=0)
    if stat > error:
        raise ValueError("stat_error must not exceed total error")
    return value, error, stat


def _targets(config, payload, *, varied=False):
    roles = _definitions(config)
    by_name = {name: {} for name in roles}
    for row in payload["moments"]:
        _fields(row, ("observable", "degree", "value", "error", "stat_error"), "moment")
        name, degree = row["observable"], row["degree"]
        if name not in roles:
            raise ValueError(f"undefined observable ID {name!r}; declared IDs: {list(roles)}")
        if roles[name] == "mixed":
            if not (isinstance(degree, list) and len(degree) == 2
                    and all(type(d) is int and 1 <= d <= config["mixed"][name]["n"] for d in degree)):
                raise ValueError(f"{name}: degree must be a pair of positive degrees within the block")
            degree = tuple(degree)
        elif type(degree) is not int or degree < 1:
            raise ValueError(f"{name}: degree must be a positive integer")
        if degree in by_name[name]:
            raise ValueError(f"duplicate moment {name} degree {degree}")
        need_error = (config.get("soft_constraints", False) or
                      (not varied and roles[name] != "mixed" and config.get("moment_selection", True)
                       and name not in config.get("fixed_orders", {})))
        entry = _read_values(row, need_error=need_error)
        by_name[name][degree] = entry

    packed = {"born": {}, "recoil": {}, "mixed": {}}
    rates = dict(payload.get("rates", {}))
    for name, ordered in by_name.items():
        role = roles[name]
        spec = config[role][name]
        if not ordered:
            raise ValueError(f"no moment rows supplied for {name}")
        with_errors = [entry[1] is not None for entry in ordered.values()]
        if any(with_errors) and not all(with_errors):
            raise ValueError(f"{name}: supply errors for every degree or omit them all")
        if role == "mixed":
            keys = sorted(ordered)
            target = {column: {degree: ordered[degree][i] for degree in keys}
                      for i, column in enumerate(("values", "errors", "stat_errors"))}
        else:
            columns = ("values", "errors", "stat_errors") if role == "born" else (
                "window_values", "window_errors", "stat_errors")
            target = {column: [ordered[degree][i] for degree in sorted(ordered)]
                      for i, column in enumerate(columns)}
            target["degrees"] = sorted(ordered)
        if not all(with_errors):
            for key in ("errors", "window_errors", "stat_errors"):
                target.pop(key, None)
        if name in config.get("fixed_orders", {}) and config["fixed_orders"][name] > len(ordered):
            raise ValueError(f"{name}: fixed order count exceeds the number of supplied degrees")
        if role == "recoil":
            lo, hi = spec["range"]
            a = (spec.get("profile") or {}).get("a", lo)
            target.update(x_match=a, x_hi=hi, soft_lo=spec.get("soft_lo", max(lo, 1e-6)))
        if role == "recoil" or (role == "mixed" and any((spec.get("profile") or {}).values())):
            if name not in rates:
                raise ValueError(f"supply the fixed-order window fraction in rates[{name!r}]")
            fraction = rates.pop(name)
            _fields(fraction, ("value", "error", "stat_error"), "window rate")
            rate, error, stat = _read_values(fraction, need_error=config.get("soft_constraints", False))
            if not 0 < rate <= 1:
                raise ValueError(f"{name}: window fraction must lie in (0,1], with stat_error <= error")
            target["rate"] = rate
            if error is not None:
                target.update(rate_stat=stat, rate_scale=np.sqrt(max(0., error**2 - stat**2)))
        elif role == "mixed":
            target["rate"] = 1.0
        packed[role][name] = target
    if rates:
        raise ValueError(f"unexpected window rates: {sorted(rates)}")
    return packed, {name: set(rows) for name, rows in by_name.items()}


def _rows(config, computed):
    rows, rates = [], {}
    for role, blocks in computed.items():
        for name, target in blocks.items():
            if role == "mixed":
                values, errors = target["values"], target["errors"]
                degrees = sorted(values)
            else:
                values = target["window_values"] if role == "recoil" else target["values"]
                errors = target["window_errors"] if role == "recoil" else target["errors"]
                degrees = list(range(len(values)))
            for i in degrees:
                degree = list(i) if role == "mixed" else i + 1
                rows.append(dict(observable=name, degree=degree, value=values[i],
                                 error=errors[i], stat_error=target["stat_errors"][i]))
            if role == "recoil" or (role == "mixed" and any((config[role][name].get("profile") or {}).values())):
                stat = target["rate_stat"]
                rates[name] = dict(value=target["rate"], stat_error=stat,
                                   error=float(np.hypot(stat, target.get("rate_scale", 0.))))
    return dict(moments=rows, **({"rates": rates} if rates else {}))


def book_targets(fixed_order, config, *, orders=None, cross_section_pb=None, selection=None):
    """Book a portable table. Scale weights share the central integration normalization."""
    declared = deepcopy(config)
    config = _joint_maps(config)
    _definitions(config)
    weight_key = config.get("weight_key", "weight")
    central = compute_fo_moments(fixed_order, config, n_max=orders, weight_key=weight_key)
    table = dict(config=declared, **_rows(config, central))
    if selection is not None:
        table["selection"] = str(selection)
    if cross_section_pb is not None:
        if _number(cross_section_pb, "cross_section_pb") <= 0:
            raise ValueError("cross_section_pb must be positive")
        table["cross_section_pb"] = cross_section_pb
    if "weight_scales" in fixed_order:
        sample = dict(fixed_order)
        scales = np.atleast_2d(sample.pop("weight_scales"))
        labels = sample.get("scale_labels", [f"scale {i}" for i in range(len(scales))])
        if np.asarray(labels).shape != (len(scales),) or len(set(map(str, labels))) != len(labels):
            raise ValueError("scale_labels must contain one distinct label per scale row, central first")
        variations = []
        central_sum = np.asarray(sample[weight_key], float).sum()
        for weights, label in zip(scales[1:], labels[1:]):
            sample[weight_key] = weights
            shifted = compute_fo_moments(sample, config, n_max=orders, weight_key=weight_key)
            member = dict(label=str(label), **_rows(config, shifted))
            if cross_section_pb is not None:
                member["cross_section_pb"] = float(cross_section_pb * (np.sum(weights) / central_sum))
            variations.append(member)
        if variations:
            table["variations"] = {"scale": variations}
    return table


def fit_targets(events, table):
    """Fit every supplied member. Return an EnsembleResult with one row per event."""
    _fields(table, ("config", "selection", "moments", "rates", "cross_section_pb", "variations", "methods"), "target table")
    config = _joint_maps(table["config"])
    central, ids = _targets(config, table)
    if "event_id" in events and np.shape(events["event_id"]) != np.shape(events[config.get("weight_key", "weight")]):
        raise ValueError("event_id must contain one ID per prior event")
    variations = {}
    for source, records in table.get("variations", {}).items():
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ValueError(f"{source}: variations must be a list of labeled members")
        labels = [record.get("label") for record in records]
        if not all(isinstance(label, str) and label.strip() for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(f"{source}: each variation needs a distinct nonempty label")
        shifts = []
        for member in records:
            _fields(member, ("label", "moments", "rates", "cross_section_pb", "prior_weight_column"), "variation")
            shifted = None
            if "moments" in member:
                shifted, varied_ids = _targets(config, member, varied=True)
                if ids != varied_ids:
                    raise ValueError(f"{source}: varied observable IDs and degrees must match the central table")
            elif "rates" in member:
                raise ValueError(f"{source}: supply the shifted moments together with their window rates")
            if "cross_section_pb" in member and "cross_section_pb" not in table:
                raise ValueError(f"{source}: member rates require a central cross_section_pb")
            if not {"moments", "cross_section_pb", "prior_weight_column"}.intersection(member):
                raise ValueError(f"{source}: supply shifted moments, a rate or a prior weight column")
            column = member.get("prior_weight_column")
            shifts.append(Variation(moments=shifted, label=member["label"],
                                    cross_section=member.get("cross_section_pb"),
                                    prior_weight=None if column is None else events[column]))
        variations[source] = shifts
    methods = table.get("methods", {})
    if set(methods).difference(variations):
        raise ValueError("each uncertainty method must name a supplied variation source")
    if "cross_section_pb" in table:
        if _number(table["cross_section_pb"], "cross_section_pb") <= 0:
            raise ValueError("cross_section_pb must be positive")
    result = upgrade_ensemble(events, central, config, variations, methods=methods, check_support=True,
                              cross_section=table.get("cross_section_pb", 1.0))
    result.unit_area_only = "cross_section_pb" not in table
    if "selection" in table:
        result.provenance["fiducial_selection"] = table["selection"]
    if result.unit_area_only:
        result.provenance["target_cross_section_pb"] = None
        for members in result.provenance["variations"].values():
            for member in members:
                member["cross_section_pb"] = None
    return result
