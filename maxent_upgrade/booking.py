"""Book signed fixed-order contributions by independent trial, then merge jobs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from ._files import _new_output
from .basis import cheb, umap
from .constraints import _window_edges, _window_fn, profile_w
from .provenance import canonical_json, json_value
from .numerical import covariance_matrix
from .targets import _definitions, _joint_maps


def _id(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positive_integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


class _Measurements:
    """The raw integrals and the denominator of each reported target."""

    def __init__(self, config, orders, degrees):
        self.config = _joint_maps(config)
        roles = _definitions(self.config)
        if orders is not None:
            orders = _positive_integer(orders, "orders")
        if not isinstance(degrees, dict) or set(degrees) - set(roles):
            raise ValueError("degrees must name declared observable IDs")
        self.degrees = {}
        self.labels = ["fiducial"]
        self.entries = [(None, None, None)]
        self.denominators = [-1]
        for role in ("born", "recoil", "mixed"):
            for name, spec in sorted(self.config.get(role, {}).items()):
                if role == "mixed":
                    n = spec["n"]
                    all_degrees = degrees.get(name, [(i, j) for i in range(1, n + 1) for j in range(1, n + 1)])
                    actual = []
                    for pair in all_degrees:
                        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                            raise ValueError(f"{name}: specify an ordered degree pair for each product")
                        pair = tuple(_positive_integer(d, name) for d in pair)
                        if max(pair) > n:
                            raise ValueError(f"{name}: a degree exceeds the configured joint order")
                        actual.append(pair)
                else:
                    default = orders or (12 if role == "born" else 20)
                    actual = [_positive_integer(d, name) for d in degrees.get(name, range(1, default + 1))]
                if not actual or len(set(actual)) != len(actual):
                    raise ValueError(f"{name}: supply a nonempty set of distinct degrees")
                self.degrees[name] = sorted(actual)
                window = role == "recoil" or (role == "mixed" and any((spec.get("profile") or {}).values()))
                denominator = 0
                if window:
                    denominator = len(self.labels)
                    self.labels.append(f"{name}:window")
                    self.entries.append((role, name, None))
                    self.denominators.append(0)
                for degree in self.degrees[name]:
                    tag = ",".join(map(str, degree)) if role == "mixed" else str(degree)
                    self.labels.append(f"{name}:T{tag}")
                    self.entries.append((role, name, degree))
                    self.denominators.append(denominator)
        self.labels = tuple(self.labels)
        self.denominators = np.asarray(self.denominators, int)

    def features(self, observables, count):
        if count == 0:
            return np.empty((0, len(self.labels)))
        columns = [np.ones(count)]
        values = {}

        def observable(name):
            if name not in values:
                value = np.asarray(observables[name], float)
                if value.shape != (count,) or not np.isfinite(value).all():
                    raise ValueError(f"{name} must have one finite observable value per contribution")
                values[name] = value
            return values[name]

        for role in ("born", "recoil", "mixed"):
            for name, spec in sorted(self.config.get(role, {}).items()):
                orders = self.degrees[name]
                if role == "mixed":
                    x, y = spec["observables"]
                    px, py = observable(x), observable(y)
                    tx = cheb(umap(px, *spec["range"][x], spec["map"][x]), max(i for i, _ in orders))
                    ty = cheb(umap(py, *spec["range"][y], spec["map"][y]), max(j for _, j in orders))
                    profile = spec.get("profile") or {}
                    window = _window_fn(px, profile.get(x)) * _window_fn(py, profile.get(y))
                    if any(profile.values()):
                        columns.append(window)
                    columns.extend(window * tx[:, i] * ty[:, j] for i, j in orders)
                else:
                    value = observable(name)
                    lo, hi = spec["range"]
                    window = 1.
                    if role == "recoil":
                        window = profile_w(value, *_window_edges(spec.get("profile"), lo, hi))
                        columns.append(window)
                        lo = spec.get("soft_lo", max(lo, 1e-6))
                    tower = cheb(umap(value, lo, hi, spec["map"]), max(orders))
                    columns.extend(window * tower[:, degree] for degree in orders)
        return np.column_stack(columns)

    def ratios(self, raw):
        if (raw[..., 0] <= 0).any() or (raw[..., self.denominators[1:]] <= 0).any():
            raise ValueError("the combined fiducial and window integrals must be positive at every supplied scale or replica")
        result = raw.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            result[..., 1:] /= raw[..., self.denominators[1:]]
        if not np.isfinite(result).all():
            raise ValueError("nonfinite normalized integrals; verify the combined fiducial and window rates")
        return result

    def ratio_covariance(self, raw, covariance):
        k = len(raw)
        jacobian = np.eye(k)
        rows = np.arange(1, k)
        denominators = self.denominators[1:]
        jacobian[rows, rows] = 1 / raw[denominators]
        jacobian[rows, denominators] -= raw[rows] / raw[denominators]**2
        transformed = jacobian @ covariance @ jacobian.T
        return (transformed + transformed.T) / 2

    def target_rows(self, values, stat, scale, covariance):
        rows, rates = [], {}
        for i, (_, name, degree) in enumerate(self.entries[1:], 1):
            entry = dict(value=float(values[i]), error=float(np.hypot(stat[i], scale[i])), stat_error=float(stat[i]))
            if degree is not None and self.denominators[i] > 0:
                entry["covariance_with_rate"] = float(covariance[i, self.denominators[i]])
            if degree is None:
                rates[name] = entry
            else:
                rows.append(dict(observable=name, degree=list(degree) if isinstance(degree, tuple) else degree, **entry))
        return dict(cross_section_pb=float(values[0]), moments=rows, **({"rates": rates} if rates else {}))


def _combine(count, mean, scatter, n, batch_mean, batch_scatter):
    """Parallel sample mean and centered cross-products, separately at each scale."""
    if count == 0:
        return n, batch_mean.copy(), batch_scatter.copy()
    total = count + n
    delta = batch_mean - mean
    scatter = scatter + batch_scatter + (n * count / total) * delta[:, :, None] * delta[:, None, :]
    return total, mean + (n / total) * delta, scatter


class MomentAccumulator:
    """One independent integration job, storing correlated rows as a single draw.

    Use signed contributions in pb *before* dividing by the number of independent
    trials. Include rejected trials as empty groups. For independent integrator
    run estimates already in pb use unit="run" and add_integrals. A group ID is
    local to its job; complete groups arrive in increasing order, exactly once.
    """
    def __init__(self, config, *, job_id, orders=None, degrees=None, scale_labels=("central",),
                 channel="inclusive", unit="group", keep_groups=False, selection=None):
        self.job_id = _id(job_id, "job_id")
        self.channel = _id(channel, "channel")
        if unit not in ("group", "run") or not isinstance(keep_groups, bool):
            raise ValueError("use unit='group' or 'run' and a boolean keep_groups")
        self.unit, self.keep_groups = unit, keep_groups
        self.selection = None if selection is None else _id(selection, "selection")
        if isinstance(scale_labels, str):
            raise ValueError("supply a sequence of scale labels, central first")
        self.scale_labels = tuple(_id(s, "scale label") for s in scale_labels)
        if not self.scale_labels or len(set(self.scale_labels)) != len(self.scale_labels):
            raise ValueError("supply distinct scale_labels with the central choice first")
        self._config = deepcopy(config)
        self._plan = _Measurements(config, orders, {} if degrees is None else degrees)
        self.raw_labels = self._plan.labels
        self.count = 0
        self._last_id = -1
        self._first_id = None
        self._mean = np.zeros((len(self.scale_labels), len(self.raw_labels)))
        self._scatter = np.zeros((*self._mean.shape, len(self.raw_labels)))
        self._ids, self._vectors = [], []

    @property
    def config(self):
        return deepcopy(self._config)

    @property
    def degrees(self):
        return deepcopy(self._plan.degrees)

    def retained_groups(self):
        """Copies of local IDs and raw vectors (units, scales, integrals), if retained."""
        if not self.keep_groups:
            raise ValueError("enable keep_groups to retain individual independent-unit vectors")
        if not self._ids:
            return np.empty(0, dtype=np.int64), np.empty((0, *self._mean.shape))
        return np.concatenate(self._ids), np.concatenate(self._vectors)

    def _weights(self, weights):
        values = np.asarray(weights, float)
        if values.ndim == 1 and len(self.scale_labels) == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] != len(self.scale_labels) or not np.isfinite(values).all():
            raise ValueError("weights must be finite (scales, contributions); first row is central")
        return values

    def fill_group(self, group_id, observables, weights):
        """One independent group, containing all real/counterterm contributions."""
        values = self._weights(weights)
        features = self._plan.features(observables, values.shape[1])
        self.add_integrals(group_id, values @ features)

    def fill_groups(self, group_ids, observables, weights):
        """Vectorized complete groups; IDs must be nondecreasing within the batch."""
        ids = np.asarray(group_ids)
        values = self._weights(weights)
        if (ids.dtype.kind not in "iu" or ids.shape != (values.shape[1],) or not len(ids)
                or np.any(ids[1:] < ids[:-1])):
            raise ValueError("give one ordered integer group ID per contribution")
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        features = self._plan.features(observables, len(ids))
        batches = [np.add.reduceat(row[:, None] * features, starts, axis=0) for row in values]
        self._add_batch(ids[starts], np.stack(batches, axis=1))

    def add_integrals(self, group_id, integrals):
        """Add one complete unit in pb with shape (scales, raw_labels)."""
        vector = np.asarray(integrals, float)
        if vector.ndim == 1 and len(self.scale_labels) == 1:
            vector = vector[None, :]
        self._add_batch(np.asarray([group_id]), vector[None, ...])

    def _add_batch(self, ids, vectors):
        if (ids.dtype.kind not in "iu" or ids[0] <= self._last_id
                or int(ids[-1]) > np.iinfo(np.int64).max):
            raise ValueError("group IDs must be nonnegative, strictly increasing integers local to this job")
        if vectors.shape != (len(ids), *self._mean.shape) or not np.isfinite(vectors).all():
            raise ValueError(f"each raw integral array must be finite with shape {self._mean.shape}")
        mean = vectors.mean(0)
        centered = vectors - mean
        scatter = np.einsum("gsi,gsj->sij", centered, centered, optimize=True)
        count, mean, scatter = _combine(self.count, self._mean, self._scatter, len(ids), mean, scatter)
        if not np.isfinite(mean).all() or not np.isfinite(scatter).all():
            raise ValueError("nonfinite integral or covariance; verify the contribution units and normalization")
        self.count, self._mean, self._scatter = count, mean, scatter
        if self._first_id is None:
            self._first_id = int(ids[0])
        self._last_id = int(ids[-1])
        if self.keep_groups:
            self._ids.append(ids.astype(np.int64, copy=True))
            self._vectors.append(vectors.copy())

    def _contract(self):
        return dict(config=self._config, degrees=self._plan.degrees, scale_labels=self.scale_labels,
                    raw_labels=self.raw_labels, selection=self.selection, unit=self.unit)

    def save(self, path):
        """Write a new mergeable NPZ. By default its size is independent of the event count."""
        if not self.count:
            raise ValueError("an empty job has no integral estimate")
        meta = dict(schema="maxent.grouped.v1", **self._contract(), job_id=self.job_id, channel=self.channel,
                    count=self.count, first=self._first_id, last=self._last_id, keep_groups=self.keep_groups)
        arrays = dict(metadata=np.asarray(canonical_json(meta)), mean=self._mean, scatter=self._scatter)
        if self.keep_groups:
            arrays.update(group_ids=np.concatenate(self._ids), group_integrals=np.concatenate(self._vectors))
        with _new_output(path, ".npz") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path):
        """Open a saved job, validating its feature ordering and numerical summary."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["schema"] != "maxent.grouped.v1":
                raise ValueError("unsupported grouped booking file")
            settings = ("config", "job_id", "channel", "degrees", "scale_labels", "unit", "keep_groups", "selection")
            book = cls(**{key: meta[key] for key in settings})
            if list(book.raw_labels) != meta["raw_labels"]:
                raise ValueError("stored raw labels differ from the observable definitions")
            book.count = _positive_integer(meta["count"], "number of independent units")
            book._first_id, book._last_id = meta["first"], meta["last"]
            if (type(book._first_id) is not int or type(book._last_id) is not int
                    or not 0 <= book._first_id <= book._last_id <= np.iinfo(np.int64).max
                    or book.count > book._last_id - book._first_id + 1):
                raise ValueError("invalid stored independent-unit IDs")
            for attr, key in (("_mean", "mean"), ("_scatter", "scatter")):
                array = data[key]
                if array.shape != getattr(book, attr).shape or not np.isfinite(array).all():
                    raise ValueError("invalid stored raw integral summary")
                setattr(book, attr, array)
            for matrix in book._scatter:
                covariance_matrix(matrix, len(book.raw_labels), "stored centered cross-products")
            if book.keep_groups:
                ids, vectors = data["group_ids"], data["group_integrals"]
                if (ids.shape != (book.count,) or ids.dtype.kind not in "iu" or (ids[1:] <= ids[:-1]).any()
                        or ids[0] != book._first_id or ids[-1] != book._last_id
                        or vectors.shape != (book.count, *book._mean.shape) or not np.isfinite(vectors).all()):
                    raise ValueError("invalid retained independent-unit vectors or IDs")
                mean = vectors.mean(0)
                centered = vectors - mean
                scatter = np.einsum("gsi,gsj->sij", centered, centered, optimize=True)
                scale = np.maximum(abs(vectors).max(0), np.finfo(float).tiny)
                close_mean = abs(mean - book._mean) <= 1e-10 * scale
                close_scatter = abs(scatter - book._scatter) <= (
                    1e-9 * np.maximum(abs(scatter), scale[:, :, None] * scale[:, None, :]))
                if not close_mean.all() or not close_scatter.all():
                    raise ValueError("retained integration vectors disagree with the saved job summary")
                book._ids, book._vectors = [ids], [vectors]
        return book


class MergedBooking:
    """Independent channel estimates, pooling the complete trial vectors within each channel."""

    def __init__(self, jobs):
        if not jobs:
            raise ValueError("supply at least one integration job")
        first = jobs[0]
        self._plan = first._plan
        self.config, self.scale_labels, self.raw_labels = first.config, first.scale_labels, first.raw_labels
        self.selection, self.unit = first.selection, first.unit
        self.jobs = []
        self.channel_counts = {}
        self._retained = {}
        self.raw_mean = np.zeros_like(first._mean)
        self._covariances = np.zeros_like(first._scatter)
        channels = {}
        seen = set()
        contract = canonical_json(first._contract())
        for job in sorted(jobs, key=lambda item: (item.channel, item.job_id)):
            if job.job_id in seen:
                raise ValueError(f"duplicate job ID {job.job_id!r}; each independent integration stream enters once")
            seen.add(job.job_id)
            if not job.count or canonical_json(job._contract()) != contract:
                raise ValueError("jobs must be nonempty and agree on maps, degrees, scales, selection and statistical unit")
            channels.setdefault(job.channel, []).append(job)
            self.jobs.append(dict(job_id=job.job_id, channel=job.channel, count=job.count,
                                  first_id=job._first_id, last_id=job._last_id))
        for channel, pieces in channels.items():
            n, mean, scatter = 0, first._mean, first._scatter
            for piece in pieces:
                n, mean, scatter = _combine(n, mean, scatter, piece.count, piece._mean, piece._scatter)
            if n < 2:
                raise ValueError(f"channel {channel!r} needs at least two independent units for numerical errors")
            self.raw_mean += mean
            self._covariances += scatter / (n * (n - 1))
            self.channel_counts[channel] = n
            if all(p.keep_groups for p in pieces):
                self._retained[channel] = np.concatenate([v[:, 0, :] for p in pieces for v in p._vectors])
        self.target_mean = self._plan.ratios(self.raw_mean)
        self._target_covariances = np.stack([self._plan.ratio_covariance(raw, cov)
                                             for raw, cov in zip(self.raw_mean, self._covariances)])
        for arr in (self.raw_mean, self._covariances, self.target_mean, self._target_covariances):
            arr.setflags(write=False)

    @property
    def raw_covariance(self):
        """Covariance of the central raw integrals in pb², with fiducial rate first."""
        return self._covariances[0]

    @property
    def target_covariance(self):
        """Central delta-method covariance, in target order: fiducial pb, then ratios."""
        return self._target_covariances[0]

    def _replicas(self, size, seed):
        rng = np.random.default_rng(seed)
        if len(self._retained) == len(self.channel_counts):
            estimates = np.zeros((size, len(self.raw_labels)))
            for group in self._retained.values():
                n = len(group)
                probabilities = np.full(n, 1 / n)
                for row in estimates:
                    row += (rng.multinomial(n, probabilities) @ group) / n
            return estimates, "bootstrap of complete independent units within each channel"
        sigma = np.sqrt(np.maximum(np.diag(self.raw_covariance), 0.))
        standard = np.where(sigma > 0, sigma, 1.)
        correlation = self.raw_covariance / standard[:, None] / standard[None, :]
        eigen, vectors = np.linalg.eigh(correlation)
        tolerance = 128 * np.finfo(float).eps * len(eigen) * max(1., eigen[-1])
        if eigen[0] < -tolerance:
            raise ValueError("the grouped raw covariance has a negative mode beyond numerical roundoff")
        root = (vectors * np.sqrt(np.where(eigen > tolerance, eigen, 0.))) @ vectors.T
        draws = (rng.standard_normal((size, len(eigen))) @ root) * sigma
        return self.raw_mean[0] + draws, "Gaussian draw of joint raw integrals"

    def targets(self, *, replicas=0, seed=1):
        """Build normalized targets, varied total rates and coherent numerical replicas.

        Quoted numerical errors use the full covariance of the normalization and
        moment numerators. Scale variations are separate coherent calculations.
        When all jobs retained their units, bootstrap whole central-scale vectors
        separately within independent channels. Otherwise draw the joint raw
        estimates using their estimated multivariate Gaussian covariance.
        """
        if (not isinstance(replicas, (int, np.integer)) or isinstance(replicas, (bool, np.bool_))
                or replicas < 0 or replicas == 1):
            raise ValueError("replicas must be zero or at least two")
        if replicas and (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0):
            raise ValueError("the replica seed must be a nonnegative integer")
        stats = np.sqrt(np.maximum(0., np.diagonal(self._target_covariances, axis1=1, axis2=2)))
        envelope = np.ptp(self.target_mean, axis=0) / 2
        central = self._plan.target_rows(self.target_mean[0], stats[0], envelope, self.target_covariance)
        table = dict(config=deepcopy(self.config), **central)
        if self.selection is not None:
            table["selection"] = self.selection
        variations = {}
        if len(self.scale_labels) > 1:
            variations["scale"] = [
                dict(label=label, **self._plan.target_rows(values, errors, envelope, covariance))
                for label, values, errors, covariance in zip(self.scale_labels[1:], self.target_mean[1:],
                                                             stats[1:], self._target_covariances[1:])]
        method = None
        if replicas:
            estimates, method = self._replicas(replicas, seed)
            shifted = self._plan.ratios(estimates)
            variations["stat"] = [dict(label=f"integration replica {i + 1}",
                                       **self._plan.target_rows(row, stats[0], envelope, self.target_covariance))
                                  for i, row in enumerate(shifted)]
        if variations:
            table["variations"] = variations
        table["integration"] = json_value(dict(
            raw_labels=self.raw_labels, scale_labels=self.scale_labels, raw_mean_pb=self.raw_mean,
            raw_covariance_pb2=self.raw_covariance, target_covariance=self.target_covariance,
            denominator_indices=self._plan.denominators, channel_counts=self.channel_counts,
            jobs=self.jobs, unit=self.unit, stat_method=method, stat_seed=int(seed) if replicas else None,
            normalization="mean independent unit in pb per channel; sum independent channels",
        ))
        return table


def merge_bookings(jobs):
    """Combine saved .npz paths or MomentAccumulator objects from independent jobs."""
    jobs = [MomentAccumulator.load(j) if isinstance(j, (str, Path)) else j for j in jobs]
    if not all(isinstance(j, MomentAccumulator) for j in jobs):
        raise TypeError("supply MomentAccumulator objects or paths to their saved NPZ files")
    return MergedBooking(jobs)
