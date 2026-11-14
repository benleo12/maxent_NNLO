"""Optional, streaming Drell--Yan HepMC adapter.

The supplied NNLO table uses pre-QED leptons. This adapter therefore requires a
sample generated with lepton QED radiation disabled; it does not infer that
setting from the record. Units are converted only in the extracted arrays.
"""
from array import array
import math
from pathlib import Path
import tempfile

import numpy as np

from .provenance import canonical_json, file_hash


DY_SELECTION = dict(mass_GeV=[66.0, 116.0], lepton_pt_min_GeV=27.0, lepton_abs_eta_max=2.5)


def _backend():
    try:
        import pyhepmc
    except ImportError as error:
        raise ImportError("HepMC support requires the optional extra: pip install '.[hepmc]'") from error
    return pyhepmc


def _prompt(particle):
    """Exclude tau/hadron decays, allowing the incoming beam hadrons."""
    pending = [particle]
    visited = set()
    while pending:
        item = pending.pop()
        if item.id in visited:
            continue
        visited.add(item.id)
        if item is not particle and item.status != 4:
            if abs(item.pid) == 15 or abs(item.pid) > 100:
                return False
        if item.production_vertex is not None:
            pending.extend(item.production_vertex.particles_in)
    return True


def dy_observables(event, *, lepton_pid=11, qed_off=False, weight_index=0):
    """Return one selected event's arrays, or None outside the DY fiducial cuts.

    `event` is one pyhepmc GenEvent. Exactly one prompt final-state opposite-sign
    pair of the requested flavour is required. Additional prompt pairs are
    rejected as ambiguous, not silently resolved by a mass-based choice.
    """
    if not qed_off:
        raise ValueError("the DY table requires pre-QED leptons: explicitly set qed_off=True for a QED-off sample")
    if lepton_pid not in (11, 13):
        raise ValueError("lepton_pid must be 11 (electron) or 13 (muon)")
    particles = [p for p in event.particles if p.status == 1 and abs(p.pid) == lepton_pid and _prompt(p)]
    if len(particles) != 2 or {p.pid for p in particles} != {-lepton_pid, lepton_pid}:
        raise ValueError(f"event {event.event_number}: expected exactly one prompt opposite-sign lepton pair")
    minus, plus = sorted(particles, key=lambda p: -p.pid)
    backend = _backend()
    factor = .001 if event.momentum_unit == backend.Units.MEV else 1.0
    p, k = [np.array([a.momentum.px, a.momentum.py, a.momentum.pz, a.momentum.e]) * factor
            for a in (minus, plus)]
    if not np.isfinite(np.r_[p, k]).all():
        raise ValueError(f"event {event.event_number}: nonfinite lepton momentum")
    pair = p + k
    pt = float(np.hypot(pair[0], pair[1]))
    mass2 = float(pair[3]**2 - pair[:3] @ pair[:3])
    if mass2 <= 0:
        return None
    mass = math.sqrt(mass2)
    pt1, pt2 = float(np.hypot(p[0], p[1])), float(np.hypot(k[0], k[1]))
    if pt1 <= 0 or pt2 <= 0:
        return None
    eta1, eta2 = math.asinh(p[2] / pt1), math.asinh(k[2] / pt2)
    if not (66 < mass < 116 and min(pt1, pt2) > 27 and max(abs(eta1), abs(eta2)) < 2.5):
        return None
    rapidity = .5 * math.log((pair[3] + pair[2]) / (pair[3] - pair[2]))
    costh = 2 * (p[2] * k[3] - p[3] * k[2]) / (mass * math.sqrt(mass2 + pt * pt))
    delta_phi = abs(math.remainder(math.atan2(p[1], p[0]) - math.atan2(k[1], k[0]), 2 * math.pi))
    phistar = math.tan((math.pi - delta_phi) / 2) / math.cosh((eta1 - eta2) / 2)
    weight = float(event.weights[weight_index])
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError(f"event {event.event_number}: nominal weight must be finite and positive")
    return dict(event_id=int(event.event_number), weight=weight, original_weight=weight,
                mll=mass, y_abs=abs(rapidity), y_ll=rapidity, pT_ll=pt,
                abs_costh=abs(costh), costh_cs=costh, phistar=phistar,
                pT_lead=max(pt1, pt2), pT_sub=min(pt1, pt2),
                lepton_minus_px=p[0], lepton_minus_py=p[1], lepton_minus_pz=p[2], lepton_minus_e=p[3],
                lepton_plus_px=k[0], lepton_plus_py=k[1], lepton_plus_pz=k[2], lepton_plus_e=k[3],
                lepton_minus_eta=eta1, lepton_plus_eta=eta2)


def collect_dy(records, *, lepton_pid=11, qed_off=False, weight_index=0,
               selected_writer=None, selected_limit=None, source=None):
    """Consume a record iterator once; retain compact arrays, not event records.

    For a generator pipe, pass pyhepmc.open(pipe, format="hepmc2") as
    `records` (or format="hepmc3"). Automatic detection needs a seekable file. An optional open
    writer receives the first `selected_limit` selected full records (all when
    None), useful for a small record round-trip. Arrays always retain every
    selected event. This function does not close readers or writers it receives.
    """
    columns, seen = {}, set()
    count, selected_written, weight_sum = 0, 0, 0.0
    last_cross_section = None
    for event in records:
        count += 1
        event_id = int(event.event_number)
        if event_id in seen:
            raise ValueError(f"duplicate event number {event_id}; assign unique IDs before combining runs")
        seen.add(event_id)
        nominal = float(event.weights[weight_index])
        if not np.isfinite(nominal) or nominal <= 0:
            raise ValueError(f"event {event_id}: nominal weight must be finite and positive")
        weight_sum += nominal
        if event.cross_section is not None:
            last_cross_section = float(event.cross_section.xsec())
        row = dy_observables(event, lepton_pid=lepton_pid, qed_off=qed_off, weight_index=weight_index)
        if row is None:
            continue
        if not columns:
            columns = {key: array("q" if key == "event_id" else "d") for key in row}
        for key, value in row.items():
            columns[key].append(value)
        if selected_writer is not None and (selected_limit is None or selected_written < selected_limit):
            selected_writer.write(event)
            selected_written += 1
    if not columns:
        raise ValueError("no events pass the released DY fiducial cuts")
    arrays = {key: np.asarray(values) for key, values in columns.items()}
    metadata = dict(schema="maxent_upgrade.hepmc.dy.v1", source=source,
                    lepton_pid=lepton_pid, lepton_qed_radiation="disabled by caller declaration",
                    selection=DY_SELECTION, units="GeV", source_records=count,
                    selected_records=len(arrays["event_id"]), source_weight_sum=weight_sum,
                    selected_weight_sum=float(arrays["weight"].sum()), weight_index=weight_index,
                    last_record_cross_section_pb=last_cross_section,
                    cross_section_note="Last record metadata; confirm final generator integration before using for normalization.",
                    selected_full_records_written=selected_written)
    arrays["record_provenance_json"] = np.array(canonical_json(metadata))
    return arrays


def extract_dy(path, *, lepton_pid=11, qed_off=False, weight_index=0):
    """Read a regular HepMC file, fingerprint it and extract selected DY arrays."""
    backend = _backend()
    source = dict(name=Path(path).name, sha256=file_hash(path))
    with backend.open(path) as records:
        return collect_dy(records, lepton_pid=lepton_pid, qed_off=qed_off,
                          weight_index=weight_index, source=source)


def attach_weights(input_path, output_path, event_ids, weights_pb, *, provenance=None,
                   name="maxent_pb_per_event", expected_source_sha256=None, variations_pb=None):
    """Write selected records with a named absolute cross-section contribution.

    Rows are joined by unique event number; their array order need not match the
    file. Original particles, vertices, weights and metadata remain in each
    selected record. Unselected events are omitted. The new weight is already
    in pb per event: do not multiply it by the generator cross section again.
    `variations_pb` optionally maps distinct weight names to complete coherent
    per-event variation vectors, in the same ID order and units as weights_pb.
    """
    backend = _backend()
    ids, values = np.asarray(event_ids), np.asarray(weights_pb, float)
    if ids.ndim != 1 or values.shape != ids.shape or not len(ids):
        raise ValueError("event IDs and weights must be matching nonempty vectors")
    if not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError("event IDs must be unique integers")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("replacement event weights must be finite and positive")
    variations = {} if variations_pb is None else dict(variations_pb)
    if not isinstance(name, str) or not name or any(not isinstance(key, str) or not key or key == name for key in variations):
        raise ValueError("new weight names must be nonempty, distinct strings")
    for key, vector in variations.items():
        vector = np.asarray(vector, float)
        if vector.shape != ids.shape or not np.isfinite(vector).all() or (vector <= 0).any():
            raise ValueError(f"variation {key}: weights must match IDs and be finite and positive")
        variations[key] = vector
    new_names = [name, *variations]
    all_values = [values, *variations.values()]
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if expected_source_sha256 is not None and file_hash(input_path) != expected_source_sha256:
        raise ValueError("event-record file hash does not match extraction provenance")
    lookup = {number: index for index, number in enumerate(ids.tolist())}
    found, runs, source_seen = set(), {}, set()
    metadata = dict(schema="maxent_upgrade.hepmc.weights.v1", units="pb per event",
                    weight_name=name, selected_records=len(ids), cross_section_pb=float(values.sum()),
                    variation_cross_sections_pb={key: float(value.sum()) for key, value in variations.items()},
                    fit=provenance)
    handle = tempfile.NamedTemporaryFile(prefix=".maxent-", suffix=".hepmc", dir=output_path.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        with backend.open(input_path) as reader, backend.open(temporary, "w") as writer:
            for event in reader:
                number = int(event.event_number)
                if number in source_seen:
                    raise ValueError(f"duplicate event number {number} in source record")
                source_seen.add(number)
                if number not in lookup:
                    continue
                old = event.run_info
                old_names = list(old.weight_names) if old is not None else []
                if not old_names:
                    old_names = [f"original_{i}" for i in range(len(event.weights))]
                if len(old_names) != len(event.weights) or set(new_names).intersection(old_names):
                    raise ValueError("original weight names are inconsistent or already contain the requested name")
                # The reader may share run_info between events; never mutate it.
                key = (tuple(old_names), str(old.tools) if old else "",
                       tuple((key, str(value)) for key, value in old.attributes.items()) if old else ())
                if key not in runs:
                    if runs:
                        raise ValueError("selected records have different run metadata; attach each run separately")
                    run = backend.GenRunInfo()
                    if old is not None:
                        run.tools = list(old.tools)
                        for attribute, value in old.attributes.items():
                            run.attributes[attribute] = str(value)
                    run.weight_names = old_names + new_names
                    run.attributes["maxent_upgrade"] = canonical_json(metadata)
                    runs[key] = run
                original_weights = list(event.weights)
                event.run_info = runs[key]
                event.weights = original_weights + [float(vector[lookup[number]]) for vector in all_values]
                writer.write(event)
                found.add(number)
        missing = set(lookup) - found
        if missing:
            raise ValueError(f"{len(missing)} requested event IDs are absent from the record")
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return metadata
