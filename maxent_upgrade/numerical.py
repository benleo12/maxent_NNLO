"""Common-run numerical inputs for fitting and independent prediction checks."""
from dataclasses import dataclass, field

import numpy as np

from .provenance import array_identity, content_hash, json_value


def covariance_matrix(value, size, name="covariance"):
    """Validate a covariance without silently dropping negative eigenmodes."""
    value = np.asarray(value, float)
    if value.shape != (size, size) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite ({size}, {size}) matrix")
    scale = max(float(np.max(np.abs(value), initial=0)), np.finfo(float).tiny)
    if not np.allclose(value, value.T, rtol=1e-10, atol=scale * 1e-12):
        raise ValueError(f"{name} must be symmetric")
    value = (value + value.T) / 2
    if size and np.linalg.eigvalsh(value).min() < -1e-10 * scale * max(size, 1):
        raise ValueError(f"{name} is not positive semidefinite")
    return value


@dataclass
class NumericalInput:
    """Named integrals sharing a *complete* central Monte Carlo covariance.

    ``mean`` has shape (integrals, coherent scales); column zero is central.
    ``covariance`` is the covariance of those raw integral estimates, not of
    individual integration points. The normalization entry is a cross section
    in pb. Scale columns are deterministic alternatives, not Gaussian replicas.
    ``sample_id`` names the statistical sample, while metadata describes its
    process, cuts, measurement functions, sources and integration settings.
    """
    labels: tuple
    mean: np.ndarray
    covariance: np.ndarray
    sample_id: str
    role: str = "training"
    normalization: int = 0
    scale_names: tuple = ()
    metadata: dict = field(default_factory=dict)
    run_ids: tuple = ()

    def __post_init__(self):
        self.labels = tuple(self.labels)
        self.mean = np.array(self.mean, dtype=float, copy=True)
        if self.mean.ndim == 1:
            self.mean = self.mean[:, None]
        if (self.mean.ndim != 2 or self.mean.shape[0] != len(self.labels)
                or not self.mean.shape[1] or not np.isfinite(self.mean).all()):
            raise ValueError("mean must contain finite named integrals by coherent scale")
        if len(set(self.labels)) != len(self.labels) or not all(self.labels):
            raise ValueError("integral labels must be unique and nonempty")
        if self.role not in {"training", "validation"} or not self.sample_id:
            raise ValueError("declare role training/validation and a nonempty sample_id")
        if not 0 <= self.normalization < len(self.labels):
            raise ValueError("normalization must index the total cross section")
        if (self.mean[self.normalization] <= 0).any():
            raise ValueError("the normalization must be positive at every scale")
        self.covariance = covariance_matrix(self.covariance, len(self.labels)).copy()
        self.scale_names = tuple(self.scale_names or (
            ["central"] + [f"scale_{i}" for i in range(1, self.mean.shape[1])]))
        if (len(self.scale_names) != self.mean.shape[1]
                or len(set(self.scale_names)) != len(self.scale_names)):
            raise ValueError("scale_names must uniquely label every coherent scale column")
        self.metadata = json_value(self.metadata)
        self.run_ids = tuple(map(str, self.run_ids))
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("run_ids must be unique")
        self.mean.setflags(write=False)
        self.covariance.setflags(write=False)

    @property
    def normalized(self):
        return self.mean / self.mean[self.normalization]

    @property
    def normalization_jacobian(self):
        raw = self.mean[:, 0]
        rate = raw[self.normalization]
        jac = np.eye(len(raw)) / rate
        jac[:, self.normalization] -= raw / rate**2
        return jac

    @property
    def normalized_covariance(self):
        jac = self.normalization_jacobian
        return jac @ self.covariance @ jac.T

    @property
    def identity(self):
        return dict(labels=list(self.labels), mean=array_identity(self.mean),
                    covariance=array_identity(self.covariance), sample_id=self.sample_id,
                    role=self.role, normalization=self.normalization,
                    scale_names=list(self.scale_names), metadata=self.metadata,
                    run_ids=list(self.run_ids))

    @property
    def fingerprint(self):
        return content_hash(self.identity)

    def selector(self, labels):
        """Rows selecting named raw integrals, in the requested order."""
        result = np.zeros((len(labels), len(self.labels)))
        for row, name in enumerate(labels):
            result[row, self.labels.index(name)] = 1
        return result

    @classmethod
    def from_runs(cls, labels, runs, *, seeds, channels, sample_id,
                  role="training", normalization=0, scale_names=(), metadata=None,
                  stream_namespaces=None):
        """Pool independent equal-budget runs within each additive channel.

        Arrays have shapes (channel, padded run, integral, scale) and
        (channel, padded run). A seed of -1 denotes an all-NaN padding row.
        Unequal channel counts are supported; partial nonfinite rows, repeated
        seeds and fewer than two actual runs in any channel are rejected.
        Seeds must be globally distinct unless separate, genuinely independent
        RNG stream namespaces are explicitly supplied per channel. Different
        precision batches for one contribution are not additive channels.
        Use the direct constructor with the complete covariance for correlated
        channels or unequal-budget run estimates.
        """
        groups = _run_groups(runs, seeds, channels, stream_namespaces)
        mean = sum(group.mean(axis=0) for _, _, group in groups)
        cov = sum(np.atleast_2d(np.cov(group[:, :, 0], rowvar=False, ddof=1)) / len(group)
                  for _, _, group in groups)
        spaces = tuple(stream_namespaces or ["rng"] * len(groups))
        ids = tuple(f"{space}:{seed}" for space, (_, actual_seeds, _) in zip(spaces, groups)
                    for seed in actual_seeds)
        meta = dict(metadata or {}, pooling="sum of channel means; covariance of means",
                    channel_counts={name: len(group) for name, _, group in groups},
                    independent_channels=True, stream_namespaces=list(spaces))
        return cls(labels, mean, cov, sample_id, role, normalization, scale_names, meta, ids)


def _run_groups(runs, seeds, channels, stream_namespaces=None):
    runs, seeds = np.asarray(runs, float), np.asarray(seeds)
    if (runs.ndim != 4 or seeds.shape != runs.shape[:2]
            or len(channels) != len(runs) or len(set(channels)) != len(channels)):
        raise ValueError("run/seed shapes and unique channel labels must agree")
    if not np.issubdtype(seeds.dtype, np.integer) or (seeds < -1).any():
        raise ValueError("seeds must be integers, with -1 as the only padding marker")
    if stream_namespaces is None:
        if len(np.unique(seeds[seeds >= 0])) != np.sum(seeds >= 0):
            raise ValueError("reused seeds need explicitly independent RNG stream namespaces")
    elif (len(stream_namespaces) != len(channels)
          or len(set(stream_namespaces)) != len(channels) or not all(stream_namespaces)):
        raise ValueError("declare one distinct independent RNG stream namespace per channel")
    groups = []
    for name, group, ids in zip(channels, runs, seeds):
        keep = ids >= 0
        if not np.isnan(group[~keep]).all() or not np.isfinite(group[keep]).all():
            raise ValueError("actual runs must be finite and padding must be all NaN")
        if keep.sum() < 2 or len(np.unique(ids[keep])) != keep.sum():
            raise ValueError("each channel needs at least two distinct run seeds")
        groups.append((name, ids[keep], group[keep]))
    return groups


def bootstrap_runs(runs, *, seeds, channels, replicas=200, seed=1, stream_namespaces=None):
    """Resample whole central-scale vectors within channels, preserving covariance.

    Returns raw integrals (replica, integral). These numerical resamples do not
    include scale variations, event sampling, or uncertainty in the model.
    """
    if replicas < 2:
        raise ValueError("at least two replicas are required")
    groups = _run_groups(runs, seeds, channels, stream_namespaces)
    rng = np.random.default_rng(seed)
    result = np.zeros((replicas, np.asarray(runs).shape[2]))
    for _, _, group in groups:
        n = len(group)
        counts = rng.multinomial(n, np.full(n, 1 / n), size=replicas)
        result += counts @ group[:, :, 0] / n
    return result
