"""Portable records of the inputs, configuration and normalization of a fit."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .reweight import DEFAULTS


def json_value(value):
    """Convert configuration and moment dictionaries to stable JSON values."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key)
            if name in result:
                raise ValueError("dictionary keys are ambiguous after serialization")
            result[name] = json_value(item)
        return result
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return json_value(value.item())
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("provenance metadata must contain finite numbers")
    return value


def canonical_json(value):
    """A deterministic representation, independent of dictionary insertion order."""
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value):
    """Fingerprint a moment table, configuration or other small structured input."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path):
    """Fingerprint a file without reading it all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_identity(values):
    """Record dtype, shape and data hash without expanding an array into JSON."""
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError("object arrays cannot be fingerprinted portably")
    contiguous = np.ascontiguousarray(array)
    return dict(dtype=array.dtype.str, shape=list(array.shape),
                sha256=hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest())


def table_identity(process, table, selection=None):
    """Identify the supplied raw table and the selection made from it."""
    selected = dict(born=None, recoil=None, mixed=None, max_orders=None, scale=None, runs=None,
                    window=None, nnlo_zj=False, pdf_member=None, qt_basis="range")
    selected.update(selection or {})
    identity = dict(identifier=Path(str(process)).name, sha256=content_hash(table),
                    selection=json_value(selected))
    for key in ("process", "order", "fiducial", "pdf", "provenance"):
        if key in table:
            identity[key] = json_value(table[key])
    return identity


def released_table_identity(process, table, selection=None, sources=()):
    """Identify the nominal table and auxiliary files used by released inputs."""
    from . import released
    selection = selection or {}
    identity = table_identity(process, table, selection)
    if process in released.PROCESSES:
        paths = {Path(released._path(process))}
    elif Path(str(process)).is_file():
        paths = {Path(process)}
    else:
        return identity
    windows = [selection.get("window") or "nominal"]
    if "window" in sources:
        windows += ["down", "up"]
    for window in windows:
        spec = released.WINDOWS.get(process, {}).get(window)
        if spec is not None:
            paths.add(Path(released._path(spec[0])))
    if "pdf" in sources or selection.get("pdf_member") is not None:
        for registry in (released.PDF_MEMBERS, released.PDF_MEMBERS_NOMINAL):
            if process in registry:
                paths.add(Path(released._path(registry[process])))
    if selection.get("nnlo_zj"):
        paths.add(Path(released._data_dir()) / released.NNLO_ZJ_FILE)
    # Re-pooling, rate evaluation and map changes read the run accumulators.
    # Retain their source identity as well as the transformed target hash.
    for path in list(paths):
        if path.suffix == ".json":
            runs = path.with_name(path.stem + "_runs.json.gz")
            if runs.is_file():
                paths.add(runs)
    identity["files"] = [dict(name=p.name, sha256=file_hash(p)) for p in sorted(paths)]
    return identity


def effective_configuration(config):
    """Record the explicit solver and measurement-function defaults."""
    configuration = {**DEFAULTS, "keep_features": False, **config}
    # These are effective defaults inside the recoil/joint construction.
    configuration = json_value(configuration)
    for spec in (configuration.get("born") or {}).values():
        spec.setdefault("map", "lin")
    for spec in (configuration.get("recoil") or {}).values():
        spec.setdefault("map", "log")
        spec.setdefault("transition", "blend")
    for spec in (configuration.get("mixed") or {}).values():
        spec.setdefault("include_rate", True)
        spec.setdefault("transition", "windowed")
        spec.setdefault("map", {})
        for observable in spec["observables"]:
            spec["map"].setdefault(observable, "log")
    return configuration


def fit_provenance(events, moments, config, result, *, cross_section=1.0,
                   normalization="pb", input_path=None, table=None):
    """Describe the exact numerical fit and its event ordering.

    Hashes identify inputs; they do not establish their physics validity. All
    supplied event arrays are fingerprinted, including event IDs when present.
    `table` may be a table_identity() record for released or custom tables.
    """
    if normalization not in {"pb", "unit_area"}:
        raise ValueError("normalization must be pb or unit_area")
    if not np.isfinite(cross_section) or cross_section <= 0:
        raise ValueError("cross_section must be finite and positive")
    configuration = effective_configuration(config)
    modules = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    record = dict(schema="maxent_upgrade.fit.v1", implementation_sha256=content_hash(modules),
                  configuration=configuration, configuration_sha256=content_hash(configuration),
                  moments_sha256=content_hash(moments), moment_table=table,
                  event_count=len(result.weights),
                  event_arrays={k: array_identity(v) for k, v in sorted(events.items())},
                  fitted_orders=json_value(result.chosen_moments),
                  normalization=normalization, target_cross_section_pb=float(cross_section),
                  output_weights=array_identity(result.weights * (cross_section if normalization == "pb" else 1)))
    if input_path is not None:
        record["input_file"] = dict(name=Path(input_path).name, sha256=file_hash(input_path))
    return record
