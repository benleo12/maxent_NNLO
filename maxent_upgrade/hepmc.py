"""Stream HepMC events into observable columns and attach fitted weights by event ID."""
from array import array
from collections.abc import Mapping
from functools import partial
import json
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


def _to_gev(event):
    return .001 if event.momentum_unit == _backend().Units.MEV else 1.


def _nominal(event, index):
    weights = event.weights
    if not weights and index == 0 and (event.run_info is None or not event.run_info.weight_names):
        return 1.
    try:
        value = float(weights[index])
    except (IndexError, TypeError) as error:
        raise ValueError(f"event {event.event_number}: generator weight index {index!r} is unavailable") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"event {event.event_number}: nominal weight must be finite and positive")
    return value


def _dy_definition(lepton_pid, qed_off):
    if not qed_off:
        raise ValueError("the DY table requires pre-QED leptons: explicitly set qed_off=True for a QED-off sample")
    if lepton_pid not in (11, 13):
        raise ValueError("lepton_pid must be 11 (electron) or 13 (muon)")
    return partial(_dy_kinematics, lepton_pid=lepton_pid)


def _dy_kinematics(event, factor, *, lepton_pid):
    particles = [p for p in event.particles if p.status == 1 and abs(p.pid) == lepton_pid and _prompt(p)]
    if len(particles) != 2 or {p.pid for p in particles} != {-lepton_pid, lepton_pid}:
        raise ValueError(f"event {event.event_number}: expected exactly one prompt opposite-sign lepton pair")
    minus, plus = sorted(particles, key=lambda p: -p.pid)
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
    return dict(mll=mass, y_abs=abs(rapidity), y_ll=rapidity, pT_ll=pt,
                abs_costh=abs(costh), costh_cs=costh, phistar=phistar,
                pT_lead=max(pt1, pt2), pT_sub=min(pt1, pt2),
                lepton_minus_px=p[0], lepton_minus_py=p[1], lepton_minus_pz=p[2], lepton_minus_e=p[3],
                lepton_plus_px=k[0], lepton_plus_py=k[1], lepton_plus_pz=k[2], lepton_plus_e=k[3],
                lepton_minus_eta=eta1, lepton_plus_eta=eta2)


def dy_observables(event, *, lepton_pid=11, qed_off=False, weight_index=0):
    """Return observables and generator weight for a selected DY event, or None."""
    measure = _dy_definition(lepton_pid, qed_off)
    row = measure(event, _to_gev(event))
    if row is None:
        return None
    nominal = _nominal(event, weight_index)
    return dict(event_id=int(event.event_number), weight=nominal, original_weight=nominal, **row)


def _collect(records, measure, weight_index, selected_writer, selected_limit, source, description):
    columns, seen = {}, set()
    event_ids, weights = array("q"), array("d")
    reserved = {"event_id", "weight", "original_weight", "record_provenance_json"}
    count, selected_written, weight_sum = 0, 0, 0.0
    last_cross_section = None
    for event in records:
        count += 1
        event_id = int(event.event_number)
        if event_id in seen:
            raise ValueError(f"duplicate event number {event_id}; assign unique IDs before combining runs")
        seen.add(event_id)
        nominal = _nominal(event, weight_index)
        weight_sum += nominal
        if event.cross_section is not None:
            last_cross_section = float(event.cross_section.xsec())
        row = measure(event, _to_gev(event))
        if row is None:
            continue
        if not isinstance(row, Mapping) or not row:
            raise ValueError(f"event {event_id}: return a nonempty mapping of observable names to scalars, or None")
        if not columns:
            if any(not isinstance(name, str) or not name or name in reserved for name in row):
                raise ValueError(f"event {event_id}: observable names must be nonempty strings distinct from {sorted(reserved)}")
            columns = {name: array("d") for name in row}
        elif row.keys() != columns.keys():
            raise ValueError(f"event {event_id}: observable columns differ from the first selected event")
        for name, value in row.items():
            if not isinstance(value, (int, float, np.integer, np.floating, np.bool_)) or not math.isfinite(value):
                raise ValueError(f"event {event_id}: observable {name!r} must be a finite real scalar")
            columns[name].append(value)
        event_ids.append(event_id)
        weights.append(nominal)
        if selected_writer is not None and (selected_limit is None or selected_written < selected_limit):
            selected_writer.write(event)
            selected_written += 1
    if not event_ids:
        raise ValueError("no events pass the supplied fiducial selection")
    nominal_weights = np.asarray(weights)
    arrays = dict(event_id=np.asarray(event_ids), weight=nominal_weights, original_weight=nominal_weights.copy())
    arrays.update({name: np.asarray(values) for name, values in columns.items()})
    metadata = dict(description, source=source, source_records=count, selected_records=len(event_ids),
                    source_weight_sum=weight_sum, selected_weight_sum=float(nominal_weights.sum()),
                    weight_index=weight_index, last_record_cross_section_pb=last_cross_section,
                    cross_section_note="Last record metadata; confirm final generator integration before using for normalization.",
                    selected_full_records_written=selected_written)
    arrays["record_provenance_json"] = np.array(canonical_json(metadata))
    return arrays


def collect_events(records, measure, *, selection=None, weight_index=0, selected_writer=None,
                   selected_limit=None, source=None):
    """Call measure(event, to_GeV) once per source event, in the original order.

    Return None from the callback to skip an event or a mapping of observable
    names to finite scalar values. Multiply original momentum components by
    to_GeV when defining observables and cuts in GeV. The input records retain
    their native units. event_id and the generator weight are added automatically.
    An empty weight vector means a unit nominal weight. A selected_writer may
    retain all accepted records or the first selected_limit; arrays cover all of them.
    Readers and writers supplied by the caller stay open.
    """
    description = dict(schema="maxent_upgrade.hepmc.observables.v1", selection=selection)
    return _collect(records, measure, weight_index, selected_writer, selected_limit, source, description)


def collect_dy(records, *, lepton_pid=11, qed_off=False, weight_index=0,
               selected_writer=None, selected_limit=None, source=None):
    """Extract the released Drell--Yan fiducial selection from an event iterator."""
    measure = _dy_definition(lepton_pid, qed_off)
    description = dict(schema="maxent_upgrade.hepmc.dy.v1", lepton_pid=lepton_pid,
                       lepton_qed_radiation="disabled by caller declaration", selection=DY_SELECTION, units="GeV")
    return _collect(records, measure, weight_index, selected_writer, selected_limit, source, description)


def _read_file(path, collect, **options):
    source = dict(name=Path(path).name, sha256=file_hash(path))
    with _backend().open(path) as records:
        return collect(records, source=source, **options)


def extract_events(path, measure, *, selection=None, weight_index=0):
    """Read a HepMC2/3 file using a user-defined measurement and fingerprint its source."""
    return _read_file(path, collect_events, measure=measure, selection=selection, weight_index=weight_index)


def extract_dy(path, *, lepton_pid=11, qed_off=False, weight_index=0):
    """Read a regular HepMC file, fingerprint it and extract selected DY arrays."""
    return _read_file(path, collect_dy, lepton_pid=lepton_pid, qed_off=qed_off, weight_index=weight_index)


def _original_cross_sections(event, count):
    """Read before changing the event's weight names (HepMC attributes are lazy)."""
    original = event.cross_section
    if original is None:
        return [(-1., 0.)] * count, -1, -1, None
    text = original.to_string()
    nominal = (float(original.xsec()), float(original.xsec_err()))
    entries = []
    for index in range(count):
        try:
            entries.append((float(original.xsec(index)), float(original.xsec_err(index))))
        except (IndexError, RuntimeError):
            # HepMC's older scalar convention has only nominal metadata.
            entries.append(nominal)
    return entries, original.accepted_events, original.attempted_events, text


def _extend_cross_sections(event, backend, original, fitted_rates):
    """Publish each new stream's rate for exactly the records being written."""
    entries, accepted, attempted, text = original
    cross_section = backend.GenCrossSection()
    event.cross_section = cross_section
    # Once attached to the resized event, the scalar setter allocates one
    # entry per weight even in pyhepmc versions without a vector setter.
    cross_section.set_cross_section(-1., 0., accepted, attempted)
    for index, (value, error) in enumerate(entries + [(rate, 0.) for rate in fitted_rates]):
        cross_section.set_xsec(index, value)
        cross_section.set_xsec_err(index, error)
    if text is not None:
        event.attributes["maxent_original_GenCrossSection"] = text


def attach_weights(input_path, output_path, event_ids, weights_pb, *, provenance=None,
                   name="maxent_pb_per_event", expected_source_sha256=None, variations_pb=None):
    """Write selected records with a named absolute cross-section contribution.

    Rows are joined by unique event number; their array order need not match the
    file. Original particles, vertices, weights and metadata remain in each
    selected record. Unselected events are omitted. The new weight is already
    in pb per event: do not multiply it by the generator cross section again.
    Each appended GenCrossSection entry is the sum of its selected weights in pb.
    Original cross-section entries remain in their original slots.
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
    fitted_rates = [float(vector.sum()) for vector in all_values]
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
                    cross_section_metadata="Appended entries are selected-record rates in pb; original entries retained, scalar original metadata broadcast to original slots, absent original rates marked -1. New error fields are zero placeholders, not propagated fit errors.",
                    fit=provenance)
    handle = tempfile.NamedTemporaryFile(prefix=".maxent-", suffix="".join(output_path.suffixes) or ".hepmc",
                                         dir=output_path.parent, delete=False)
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
                original_weights = list(event.weights)
                if not old_names and not original_weights:
                    old_names, original_weights = ["nominal"], [1.]
                elif not old_names:
                    old_names = [f"original_{i}" for i in range(len(original_weights))]
                if len(old_names) != len(original_weights) or set(new_names).intersection(old_names):
                    raise ValueError("original weight names are inconsistent or already contain the requested name")
                original_cross_sections = _original_cross_sections(event, len(original_weights))
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
                event.run_info = runs[key]
                event.weights = original_weights + [float(vector[lookup[number]]) for vector in all_values]
                _extend_cross_sections(event, backend, original_cross_sections, fitted_rates)
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


def attach_fit(input_path, fitted_path, output_path):
    """Attach the central and every coherent pb weight from a saved fit to HepMC.

    When extract_events or extract_dy recorded a file hash, verify it before
    matching records by their original IDs. Member names use a stable index;
    their physical labels are retained in the output's run provenance.
    """
    with np.load(fitted_path, allow_pickle=False) as fit:
        if "weight_normalization" not in fit or fit["weight_normalization"].item() != "pb":
            raise ValueError("attach_fit requires replacement weights in pb; supply the fiducial cross section")
        if "event_id" not in fit:
            raise ValueError("attach_fit requires the original HepMC event_id for each fitted event")
        record = json.loads(fit["record_provenance_json"].item()) if "record_provenance_json" in fit else {}
        source = record.get("source") or {}
        digest = source.get("sha256") if isinstance(source, dict) else None
        provenance = json.loads(fit["provenance_json"].item()) if "provenance_json" in fit else {}
        variations, labels = {}, {}
        for group in fit.get("uncertainty_sources", []):
            names, rows = fit["labels_" + group], fit["weight_" + group]
            if len(names) != len(rows) or rows.ndim != 2:
                raise ValueError(f"uncertainty {group}: expected one label per event-weight vector")
            for i, (label, row) in enumerate(zip(names, rows), 1):
                name = f"maxent_{group}_{i:03d}_pb_per_event"
                variations[name] = row
                labels[name] = str(label)
        provenance.update(event_record=record, hepmc_weight_labels=labels)
        return attach_weights(input_path, output_path, fit["event_id"], fit["weight_upgraded"],
                              expected_source_sha256=digest, variations_pb=variations, provenance=provenance)
