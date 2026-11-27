"""Book user-defined moments in NNLOJET and pool independent signed runs."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import json
import math
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from ._files import _new_output
from .booking import MomentAccumulator, _Measurements, merge_bookings
from .constraints import _window_edges
from .provenance import canonical_json, content_hash, file_hash
from .targets import _fields


def _block(card, name):
    pattern = rf"^[ \t]*{name}\b([^\n]*)\n(.*?)^[ \t]*END_{name}\b[^\n]*"
    matches = list(re.finditer(pattern, card, re.M | re.S | re.I))
    if len(matches) != 1:
        raise ValueError(f"the NNLOJET run card must contain one {name} block")
    return matches[0]


def _axis_row(native, degree, interval, mapping, window, fallback):
    lo, hi = interval
    mass = width = 0.
    if isinstance(mapping, (list, tuple)):
        _, mass, width = mapping
        code = 3
    else:
        code = {"lin": 1, "log": 2}[mapping]
    a, b, c, d = window or (0., 0., 0., 0.)
    params = (lo, hi, mass, width, a, b, c or 0., d or 0.)
    if fallback is not None:
        params += (fallback,)
    return f"{native} {degree} {code} " + " ".join(format(float(x), ".17g") for x in params)


def _measurements(setup):
    plan = _Measurements(setup["config"], setup.get("orders"), setup.get("degrees", {}))
    axes = setup["axes"]
    if not isinstance(axes, dict):
        raise ValueError("axes must map event column names to registered NNLOJET observables")
    used = set()
    rows = []
    for role, name, degree in plan.entries[1:]:
        spec = plan.config[role][name]
        obs = spec["observables"] if role == "mixed" else [name]
        powers = [0] * len(obs) if degree is None else (degree if role == "mixed" else [degree])
        rows.append(str(len(obs)))
        for axis, power in zip(obs, powers):
            used.add(axis)
            native = axes.get(axis)
            fallback = None
            if isinstance(native, dict):
                _fields(native, ("observable", "undefined"), f"NNLOJET axis {axis}")
                fallback = native.get("undefined")
                try:
                    valid = type(fallback) in (float, int) and math.isfinite(fallback)
                except OverflowError:
                    valid = False
                if not valid:
                    raise ValueError(f"{axis}: explicitly supply a finite number for undefined values")
                native = native.get("observable")
            if not isinstance(native, str) or not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_+-]{0,31}", native):
                raise ValueError(f"axes must identify a registered NNLOJET observable for {axis!r}")
            window = None
            if role == "mixed":
                limits, mapping = spec["range"][axis], spec["map"][axis]
                prof = (spec.get("profile") or {}).get(axis)
                if prof:
                    window = (prof["a"], prof["b"], prof.get("c"), prof.get("d"))
            else:
                limits, mapping = spec["range"], spec["map"]
                if role == "recoil":
                    lo, hi = limits
                    window = _window_edges(spec.get("profile"), lo, hi)
                    limits = (spec.get("soft_lo", max(lo, 1e-6)), hi)
            rows.append(_axis_row(native, power, limits, mapping, window, fallback))
    if set(axes) != used:
        raise ValueError("axes must map exactly the event columns in config to their NNLOJET names")
    return plan.labels, "\n".join([str(len(plan.entries) - 1), *rows]) + "\n"


def _logical_lines(text):
    content = re.sub(r"![^\n]*", "", text)
    return [line.strip() for line in re.sub(r"&\s*\n\s*&", "", content).splitlines() if line.strip()]


def _cards(template, channels, count):
    if re.search(r"\bmxrow[0-9]+\b", "\n".join(_logical_lines(template)), re.I):
        raise ValueError("the run card already contains reserved mxrow histogram names")
    hist = _block(template, "HISTOGRAMS")
    lines = ["  cross > mxrow0"] + [f"  cross > mxrow{i} fac=mxmoment{i}" for i in range(1, count)]
    text = template[:hist.end(2)] + "\n".join(lines) + "\n" + template[hist.end(2):]
    block = _block(text, "CHANNELS")
    if channels is None:
        entries = " ".join(_logical_lines(block[2]))
        channels = re.sub(r"\s+\[", "[", entries).split()
    if (not isinstance(channels, list) or not channels
            or not all(isinstance(ch, str) and re.fullmatch(r"[A-Za-z0-9_+-]+(?:\[[\w,+.-]+\])?", ch)
                       for ch in channels)
            or len({ch.lower() for ch in channels}) != len(channels)):
        raise ValueError("supply distinct NNLOJET channel names, one per independent contribution")
    return {ch: text[:block.start(2)] + f"  {ch}\n" + text[block.end(2):] for ch in channels}


def _dependencies(spec, base):
    """Map each private run-directory filename to a source file and its expected digest."""
    if isinstance(spec, list) and all(isinstance(x, str) and x for x in spec):
        pairs = [(Path(x).name if Path(x).is_absolute() else x, x) for x in spec]
    elif isinstance(spec, dict):
        pairs = list(spec.items())
    else:
        raise ValueError("dependencies must be a list of files or a mapping of staged names to source files")
    staged = {}
    for dest, source in pairs:
        if not isinstance(dest, str) or not isinstance(source, str) or not source:
            raise ValueError("each dependency needs a staged filename and a source filename")
        dest = Path(dest)
        if (dest.is_absolute() or not dest.parts or ".." in dest.parts
                or dest.parts[0] in {"moments.run", "maxent_measures.dat", "nnlojet.log", "booking.npz", "booking.sha256"}
                or re.search(r"\.mxrow\d+\.s\d+\.dat$", dest.name, re.I)
                or any(dest == other or dest in other.parents or other in dest.parents for other in staged)):
            raise ValueError(f"invalid or conflicting staged dependency {str(dest)!r}")
        path = (base / source).resolve()
        if not path.is_file():
            raise ValueError(f"missing NNLOJET dependency file {path}")
        staged[dest] = (path, file_hash(path))
    return staged


def _runtime_files(executable, template, env):
    """Fingerprint the installed matrix elements and the PDF members named on the card."""
    def directories(var):
        return [Path(x) for x in env.get(var, "").split(os.pathsep) if x]

    prefix = executable.parent.parent
    libdirs = directories("DYLD_LIBRARY_PATH") + directories("LD_LIBRARY_PATH") + [prefix / "lib", executable.parent]
    pdfdirs = directories("LHAPDF_DATA_PATH") + directories("LHAPATH") + [prefix / "share/LHAPDF"]
    pdfs = re.findall(r"\bPDF[12]?\s*=\s*([\w.+-]+)(?:\s*\[\s*(\d+)\s*\])?",
                      "\n".join(_logical_lines(template)), re.I)
    config = shutil.which("lhapdf-config")
    if config and pdfs:
        for flag, dirs in (("--datadir", pdfdirs), ("--libdir", libdirs)):
            result = subprocess.run([config, flag], text=True, capture_output=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                dirs.append(Path(result.stdout.strip()))
    files = set()
    for directory in dict.fromkeys(libdirs):
        for prefix_glob in ("libnnlojet_*", "libLHAPDF*"):
            files.update(p.resolve() for p in directory.glob(prefix_glob)
                         if re.search(r"(?:\.dylib|\.so(?:\.\d+)*)$", p.name) and p.is_file())
    unresolved = []
    for name, member in sorted(set(pdfs)):
        for relative in (f"{name}/{name}.info", f"{name}/{name}_{int(member or 0):04d}.dat"):
            path = next((p / relative for p in pdfdirs if (p / relative).is_file()), None)
            if path is None:
                unresolved.append(relative)
            else:
                files.add(path.resolve())
    if pdfs:
        files.update((d / "lhapdf.conf").resolve() for d in pdfdirs if (d / "lhapdf.conf").is_file())
    return {"files": {str(p): file_hash(p) for p in sorted(files)}, "unresolved_pdf_files": unresolved}


def _raw_integrals(folder, seed, count, to_pb, scales, single_file=None):
    """Read native zero-dimensional totals in declared scale order; signed integrals are allowed."""
    def read_one(lines, path):
        def header(name):
            values = [line.split(":", 1)[1].strip() for line in lines if line.startswith("#" + name + ":")]
            if len(values) != 1:
                raise ValueError(f"{path}: missing or ambiguous #{name} in integrated output")
            return values[0]

        if header("nx") != "0":
            raise ValueError(f"{path}: expected an integrated NNLOJET cross section (#nx: 0)")
        evaluations = header("neval")
        if not evaluations.isdigit() or int(evaluations) < 1:
            raise ValueError(f"{path}: expected a positive #neval for the completed production run")
        names = header("labels").split()
        expected = []
        for i in range(1, scales + 1):
            expected.extend([f"tot_scale{i:02d}[{2 * i - 1}]", f"tot_scale{i:02d}_Err[{2 * i}]"])
        if names != expected:
            raise ValueError(f"{path}: native scale columns differ from the {scales} declared scale choices")
        numeric = [line.split() for line in lines if line and not line.startswith("#")]
        if len(numeric) != 1 or len(numeric[0]) != len(names):
            raise ValueError(f"{path}: expected one value/error pair per scale")
        row = np.asarray(numeric[0], dtype=float)
        if not np.isfinite(row).all() or (row[1::2] < 0).any():
            raise ValueError(f"{path}: native values must be finite and integration errors nonnegative")
        return int(evaluations), row[::2] * to_pb

    def sources(name):
        suffix = f".{name}.s{seed}.dat"
        matches = list(folder.glob(f"*.{name}.s{seed}.dat"))
        if not matches:
            raise ValueError(f"missing integrated output {name} for seed {seed} in {folder}")
        return {p.name[:-len(suffix)]: p for p in matches}

    # One native channel can expand into several contributions (e.g. two antenna
    # subtraction regions). Sum their correlated vectors within each independent seed.
    components = sources(single_file or "mxrow0")
    result = []
    if single_file:
        for path in sorted(components.values()):
            chunks = re.split(r"^#name:[ \t]*(\S+)[ \t]*$", path.read_text(), flags=re.M)
            groups = {}
            for name, chunk in zip(chunks[1::2], chunks[2::2]):
                if name in groups:
                    raise ValueError(f"{path}: duplicate histogram {name}")
                groups[name] = chunk
            missing = [f"mxrow{i}" for i in range(count) if f"mxrow{i}" not in groups]
            if missing:
                raise ValueError(f"{path}: missing integrated outputs {missing}")
            result.append([read_one([s.strip() for s in groups[f"mxrow{i}"].splitlines()], path)
                           for i in range(count)])
    else:
        files = [components] + [sources(f"mxrow{i}") for i in range(1, count)]
        if any(group.keys() != components.keys() for group in files):
            raise ValueError(f"inconsistent native contributions among the requested moments for seed {seed}")
        for key in sorted(components):
            result.append([read_one([s.strip() for s in group[key].read_text().splitlines()], group[key])
                           for group in files])
    for rows in result:
        if len({n for n, _ in rows}) != 1:
            raise ValueError(f"fiducial and moment integrals used different production counts for seed {seed}")
    raw = sum(np.stack([values for _, values in rows], axis=1) for rows in result)
    return raw, sorted(components)


@contextmanager
def _job_lock(path):
    # NNLOJET runs on POSIX; the kernel releases this advisory lock if a worker dies.
    import fcntl
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _write_once(path, data):
    try:
        with _new_output(path, ".json") as stream:
            stream.write(data)
    except (FileExistsError, ValueError):
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"cached calculation metadata or target table differs: {path}") from None


def calculate(setup_path, setup, *, binary, directory, workers=1, replicas=0, seed=1):
    """Execute a physics run card with extra moment bookings; return a reusable target JSON path."""
    _fields(setup, ("runcard", "axes", "config", "orders", "degrees", "selection", "channels",
                    "seeds", "scale_labels", "dependencies", "output_unit"), "NNLOJET setup")
    missing = {"runcard", "axes", "config", "seeds"}.difference(setup)
    if missing:
        raise ValueError(f"NNLOJET setup is missing {', '.join(sorted(missing))}")
    if (any(type(n) is not int for n in (workers, replicas, seed)) or workers < 1
            or replicas < 0 or replicas == 1 or seed < 0):
        raise ValueError("use positive jobs, a nonnegative seed and either zero or at least two replicas")
    unit = setup.get("output_unit", "fb")
    if unit not in ("fb", "pb", "nb"):
        raise ValueError("declare the native NNLOJET cross-section unit as fb, pb or nb")
    to_pb = {"fb": .001, "pb": 1., "nb": 1000.}[unit]
    executable = shutil.which(str(binary))
    if executable is None:
        raise ValueError(f"cannot find NNLOJET executable {binary!r}; specify --nnlojet or set NNLOJET")
    executable = Path(executable).resolve()
    base = Path(setup_path).resolve().parent
    if not isinstance(setup["runcard"], str) or not (base / setup["runcard"]).is_file():
        raise ValueError(f"missing NNLOJET run card {setup['runcard']!r} relative to {base}")
    template = (base / setup["runcard"]).read_text()
    labels, measures = _measurements(setup)
    cards = _cards(template, setup.get("channels"), len(labels))
    single = _block(template, "HISTOGRAMS")[1].split("!")[0].strip()
    if single and not re.fullmatch(r">\s*[\w.+-]+", single):
        raise ValueError("invalid HISTOGRAMS output name on the run card")
    single = single[1:].strip() if single else None
    scale_choices = _logical_lines(_block(template, "SCALES")[2])
    nscales = len(scale_choices)
    scale_labels = setup.get("scale_labels", scale_choices)
    if (not isinstance(scale_labels, list) or not nscales or len(scale_labels) != nscales
            or not all(isinstance(label, str) and label.strip() for label in scale_labels)
            or len(set(scale_labels)) != nscales):
        raise ValueError("supply a distinct scale label for each SCALES entry, central first")
    selection = setup.get("selection")
    if selection is not None and (not isinstance(selection, str) or not selection.strip()):
        raise ValueError("selection must describe the fiducial cuts in a nonempty string")
    seeds = setup["seeds"]
    if (not isinstance(seeds, list) or len(seeds) < 2
            or not all(type(s) is int and s > 0 for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must specify at least two distinct positive independent seeds per channel")
    stride = max(seeds) - min(seeds) + 1
    jobs = [(ch, s + k * stride) for k, ch in enumerate(cards) for s in seeds]
    if max(s for _, s in jobs) > 999999:
        raise ValueError("generated channel seeds exceed the NNLOJET v1.0.2 limit of 999999")
    staged = _dependencies(setup.get("dependencies", []), base)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    path_variables = ("LHAPDF_DATA_PATH", "LHAPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
    for var in path_variables:
        if env.get(var):
            env[var] = os.pathsep.join(str(Path(x).absolute()) if x else "" for x in env[var].split(os.pathsep))
    manifest = dict(
        format_version=2, setup=setup, executable=str(executable), executable_sha256=file_hash(executable),
        cards=cards, measures=measures, raw_labels=labels, runtime=_runtime_files(executable, template, env),
        dependencies={str(dest): {"source": str(path), "sha256": digest} for dest, (path, digest) in staged.items()},
        environment={key: env.get(key) for key in (*path_variables, "OMP_NUM_THREADS")})
    work = Path(directory) / content_hash(manifest)
    work.mkdir(parents=True, exist_ok=True)
    _write_once(work / "calculation.json", canonical_json(manifest).encode())

    def integrate(job):
        channel, run_seed = job
        name = f"{channel}-{run_seed}"
        done = work / f"{name}.npz"
        with _job_lock(work / f"{name}.lock"):
            receipt = work / f"{name}.json"
            if done.is_file() and receipt.is_file():
                try:
                    meta = json.loads(receipt.read_text())
                    if meta["sha256"] == file_hash(done) and meta["components"]:
                        return done, meta["components"]
                except (ValueError, KeyError, TypeError):
                    pass  # An interrupted cache write is recoverable from this independent run.
            folder = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=work))
            for dest, (path, expected) in staged.items():
                local = folder / dest
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, local)
                if file_hash(local) != expected:
                    raise ValueError(f"dependency {path} changed while staging the calculation; restart it")
            (folder / "moments.run").write_text(cards[channel])
            (folder / "maxent_measures.dat").write_text(measures)
            log_path = folder / "nnlojet.log"
            with log_path.open("w") as log:
                proc = subprocess.run([str(executable), "--run", "moments.run", "--iseed", str(run_seed)],
                                      cwd=folder, stdout=log, stderr=subprocess.STDOUT, env=env)
            if proc.returncode:
                raise RuntimeError(f"NNLOJET failed for {channel}, seed {run_seed}; see {log_path}")
            try:
                raw, components = _raw_integrals(folder, run_seed, len(labels), to_pb, nscales, single)
                # Fortran STOP can return zero after an intermediate iteration wrote histograms.
                with log_path.open("rb") as log:
                    log.seek(max(0, log_path.stat().st_size - 16384))
                    if not re.search(rb"^\s*Elapsed time\s*=", log.read(), re.M):
                        raise ValueError("missing the NNLOJET completion record")
            except (ValueError, OSError) as exc:
                raise ValueError(f"invalid NNLOJET output for {name}: {exc}; see {log_path}") from exc
            acc = MomentAccumulator(setup["config"], orders=setup.get("orders"), degrees=setup.get("degrees"),
                                    scale_labels=scale_labels, selection=selection, channel=channel,
                                    job_id=name, unit="run", keep_groups=True)
            acc.add_integrals(run_seed, raw)
            saved = folder / "booking.npz"
            acc.save(saved)
            checksum = folder / "booking.sha256"
            checksum.write_text(canonical_json(dict(sha256=file_hash(saved), components=components)))
            os.replace(saved, done)
            os.replace(checksum, receipt)
        return done, components

    print(f"NNLOJET: {len(jobs)} channel/seed runs in {work}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(integrate, jobs))
    native_channels = {}
    for (channel, _), (_, components) in zip(jobs, results):
        if native_channels.setdefault(channel, components) != components:
            raise ValueError(f"native contribution names changed across seeds for {channel}")
    owners = {}
    for channel, components in native_channels.items():
        for name in components:
            owner = owners.setdefault(name, channel)
            if owner != channel:
                raise ValueError(f"native contribution {name} is counted in both {owner} and {channel}")
    table = merge_bookings([path for path, _ in results]).targets(replicas=replicas, seed=seed)
    out = work / f"targets-{replicas}-{seed}.json"
    _write_once(out, (json.dumps(table, indent=2, allow_nan=False) + "\n").encode())
    return out
