"""Fit, diagnose, freeze and independently validate declared constraint blocks.

This is the explicit-feature interface to the same solver used by ``upgrade``.
All statistical responses are first order about the central fit. The empirical
prior is treated as independent weighted rows; fixed quotas or correlated event
records require an externally validated sampling calculation instead.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import math

import numpy as np

from .diagnostics import weight_diagnostics
from .numerical import NumericalInput, covariance_matrix
from .provenance import array_identity, canonical_json, content_hash, file_hash, json_value
from .result import UpgradeResult
from .solver import _newton_maxent


@dataclass
class ConstraintBlock:
    """Named event features and targets ``offset + matrix @ normalized_input``.

    ``features`` has shape (events, features), and ``matrix`` has shape
    (features, raw integrals). Use ``input.selector(names)`` for directly booked
    features. ``prior_derivative[i,k]`` is d target_k / d p_i for normalized
    prior weights, evaluated at the central input; it defaults to zero. It is
    needed for targets containing a prior-dependent continuation/blend. Features
    themselves must be fixed under resampling. Metadata records their physical
    definition, units, maps and windows; the numerical arrays are also hashed.
    """
    name: str
    features: np.ndarray
    matrix: np.ndarray
    labels: tuple
    offset: np.ndarray = None
    prior_derivative: np.ndarray = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.features = np.asarray(self.features, float)
        self.matrix = np.asarray(self.matrix, float)
        self.labels = tuple(self.labels)
        if (not self.name or self.features.ndim != 2 or self.matrix.ndim != 2
                or self.features.shape[1] != len(self.labels)
                or self.matrix.shape[0] != len(self.labels)
                or len(set(self.labels)) != len(self.labels)):
            raise ValueError("block name, labels, feature columns and target rows must agree")
        self.offset = (np.zeros(len(self.labels)) if self.offset is None
                       else np.asarray(self.offset, float))
        self.prior_derivative = (np.zeros_like(self.features) if self.prior_derivative is None
                                 else np.asarray(self.prior_derivative, float))
        if self.offset.shape != (len(self.labels),) or self.prior_derivative.shape != self.features.shape:
            raise ValueError("offset or prior derivative has the wrong shape")
        if not all(np.isfinite(x).all() for x in (
                self.features, self.matrix, self.offset, self.prior_derivative)):
            raise ValueError("features and target definitions must be finite")
        self.metadata = json_value(self.metadata)

    @property
    def identity(self):
        return dict(name=self.name, labels=list(self.labels), metadata=self.metadata,
                    **{key: array_identity(getattr(self, key)) for key in
                       ("features", "matrix", "offset", "prior_derivative")})

    @classmethod
    def from_input(cls, name, features, data, *, labels, metadata=None):
        """Build a directly booked block without manually constructing selectors."""
        return cls(name, features, data.selector(labels), tuple(labels), metadata=metadata or {})


@dataclass
class Prediction:
    """Coherent predictions with reusable common-input and common-event responses.

    ``mean`` is (scale, observable). ``sources`` retain raw-integral Jacobians;
    ``priors`` retain per-row influence vectors. Shared identifiers are paired
    before forming a covariance. Different sources are assumed independent.
    No theoretical uncertainty is inferred from a numerical covariance.
    """
    mean: np.ndarray
    labels: tuple
    scale_names: tuple
    sources: dict
    priors: dict
    provenance: dict = field(default_factory=dict)
    frozen: tuple = ()

    @property
    def covariance_numerical(self):
        out = np.zeros((len(self.labels), len(self.labels)))
        for source in self.sources.values():
            jac = source["jacobian"]
            out += jac @ source["covariance"] @ jac.T
        return (out + out.T) / 2

    @property
    def covariance_prior(self):
        out = np.zeros((len(self.labels), len(self.labels)))
        for prior in self.priors.values():
            out += prior["influence"].T @ prior["influence"]
        return (out + out.T) / 2

    @property
    def covariance(self):
        return self.covariance_numerical + self.covariance_prior

    def transform(self, function, jacobian, *, labels, definition):
        """Apply a declared observable transformation and its central Jacobian.

        ``function`` acts on all scale rows; ``jacobian`` is (new, old). This
        also supports ratios, acceptances and charge asymmetries after stacking
        their underlying predictions. Covariance propagation remains linear.
        """
        mean = np.asarray(function(self.mean), float)
        jac = np.asarray(jacobian, float)
        if (mean.shape != (len(self.scale_names), len(labels))
                or jac.shape != (len(labels), len(self.labels))
                or not np.isfinite(mean).all() or not np.isfinite(jac).all()):
            raise ValueError("transformed values/Jacobian must have finite matching dimensions")
        sources = {k: dict(v, jacobian=jac @ v["jacobian"]) for k, v in self.sources.items()}
        priors = {k: dict(v, influence=v["influence"] @ jac.T) for k, v in self.priors.items()}
        return Prediction(mean, tuple(labels), self.scale_names, sources, priors,
                          dict(operation=definition, input=self.provenance), self.frozen)

    def difference(self, other):
        """Paired difference, retaining shared numerical and event fluctuations."""
        if self.labels != other.labels:
            raise ValueError("paired predictions must have identical observable labels")
        joint = stack_predictions([self, other])
        n = len(self.labels)
        jac = np.concatenate([np.eye(n), -np.eye(n)], axis=1)
        return joint.transform(lambda a: a[:, :n] - a[:, n:], jac, labels=self.labels,
                               definition="paired model difference")

    def numerical_replicas(self, raw_replicas):
        """Evaluate the linear response on common raw-integral resamples.

        ``raw_replicas`` maps every numerical source fingerprint to its array
        (replica, integral). Reuse identical draws across model comparisons.
        """
        if set(raw_replicas) != set(self.sources):
            raise ValueError("supply a common replica array for every numerical source")
        count = {len(a) for a in raw_replicas.values()}
        if len(count) != 1:
            raise ValueError("all numerical sources must use the same replica count")
        out = np.tile(self.mean[0], (count.pop(), 1))
        for key, source in self.sources.items():
            raw = np.asarray(raw_replicas[key], float)
            if raw.shape[1:] != source["central_raw"].shape or not np.isfinite(raw).all():
                raise ValueError("raw replica dimensions/values do not match the source")
            out += (raw - source["central_raw"]) @ source["jacobian"].T
        return out

    def summary(self):
        return json_value(dict(labels=self.labels, scale_names=self.scale_names,
                               mean=self.mean, covariance_numerical=self.covariance_numerical,
                               covariance_prior=self.covariance_prior,
                               provenance=self.provenance, frozen=self.frozen,
                               uncertainty="first-order common-input and independent-row influence covariance; scales separate"))


def stack_predictions(predictions):
    """Join predictions, pairing shared inputs/events and retaining cross-covariance."""
    if not predictions or any(p.scale_names != predictions[0].scale_names for p in predictions):
        raise ValueError("predictions must have identical coherent scale ordering")
    sizes = [len(p.labels) for p in predictions]
    total = sum(sizes)
    sources, priors, sample_sources, prior_samples = {}, {}, {}, {}
    start = 0
    for prediction, size in zip(predictions, sizes):
        for key, source in prediction.sources.items():
            sample = source["identity"]["sample_id"]
            if sample in sample_sources and sample_sources[sample] != key:
                raise ValueError("the same numerical sample has different definitions; provide its full common vector")
            sample_sources[sample] = key
            if key not in sources:
                run_ids = set(source["identity"]["run_ids"])
                if any(run_ids & set(v["identity"]["run_ids"]) for v in sources.values()):
                    raise ValueError("distinct numerical samples overlap in run IDs; provide common covariance")
                sources[key] = dict(source, jacobian=np.zeros((total, source["jacobian"].shape[1])))
            elif not np.array_equal(sources[key]["covariance"], source["covariance"]):
                raise ValueError("the same numerical source has inconsistent covariance")
            sources[key]["jacobian"][start:start + size] = source["jacobian"]
        for key, prior in prediction.priors.items():
            sample = prior["identity"]["sample_id"]
            if sample in prior_samples and prior_samples[sample] != key:
                raise ValueError("the same prior sample has different event ordering/subsets; align IDs first")
            prior_samples[sample] = key
            if key not in priors:
                priors[key] = dict(prior, influence=np.zeros((len(prior["influence"]), total)))
            priors[key]["influence"][:, start:start + size] = prior["influence"]
        start += size
    return Prediction(np.concatenate([p.mean for p in predictions], axis=1),
                      tuple(label for p in predictions for label in p.labels),
                      predictions[0].scale_names, sources, priors,
                      dict(operation="stack", inputs=[p.provenance for p in predictions]),
                      tuple(sorted({f for p in predictions for f in p.frozen})))


def _source(data, jacobian):
    return {data.fingerprint: dict(jacobian=jacobian, covariance=data.covariance,
                                   central_raw=data.mean[:, 0], identity=data.identity)}


@dataclass
class ConstraintFit:
    """The existing UpgradeResult, coherent scale solves, and response information."""
    central: UpgradeResult
    scale_weights: np.ndarray
    scale_lam: np.ndarray
    data: NumericalInput
    blocks: tuple
    p: np.ndarray
    event_ids: np.ndarray
    prior_id: str
    l2: float
    phi: np.ndarray
    matrix: np.ndarray
    offset: np.ndarray
    prior_derivative: np.ndarray
    labels: tuple
    frozen_id: str = None
    freeze_record: dict = None
    fit_configuration_sha256: str = None
    _geometry: dict = field(default=None, repr=False)

    @property
    def weights(self):
        return self.central.weights

    @property
    def prior_identity(self):
        return dict(sample_id=self.prior_id, event_ids=array_identity(self.event_ids),
                    normalized_weights=array_identity(self.p),
                    sampling="independent weighted rows; Poisson influence covariance")

    @property
    def prior_key(self):
        identity = self.prior_identity.copy()
        identity.pop("normalized_weights")
        return content_hash(identity)

    @property
    def configuration(self):
        return dict(schema="maxent_upgrade.constraint-study.v1", input=self.data.identity,
                    prior=self.prior_identity, blocks=[b.identity for b in self.blocks],
                    l2=self.l2, objective="existing prior-standardized hard-constraint dual",
                    normalization="raw normalization integral in pb")

    def _response(self, observables):
        h = np.asarray(observables, float)
        if h.ndim == 1:
            h = h[:, None]
        if h.ndim != 2 or len(h) != len(self.p) or not np.isfinite(h).all():
            raise ValueError("observables must be finite event-by-observable columns")
        q = self.weights
        hc = h - q @ h
        if not self.phi.shape[1]:
            return h, np.zeros((h.shape[1], 0)), q[:, None] * hc
        if self._geometry is None:
            posterior_mean, prior_mean = q @ self.phi, self.p @ self.phi
            fc = self.phi - posterior_mean
            variance = self.p @ (self.phi - prior_mean)**2
            hessian = (q[:, None] * fc).T @ fc + self.l2 * np.diag(variance)
            self._geometry = dict(posterior_mean=posterior_mean, prior_mean=prior_mean,
                                  variance=variance, hessian=hessian)
        geometry = self._geometry
        variance, hessian = geometry["variance"], geometry["hessian"]
        cross = (q[:, None] * hc).T @ (self.phi - geometry["posterior_mean"])
        if self.l2 == 0:
            spectrum = np.linalg.eigvalsh(hessian)
            if spectrum.min() <= 1e-12 * max(spectrum.max(), np.finfo(float).tiny):
                raise ValueError("response is not identifiable at l2=0; remove dependent features or declare a positive ridge")
        try:
            response = np.linalg.solve(hessian, cross.T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("fit-response Hessian is singular; remove dependent/unsupported features") from exc
        physical_lam = self.scale_lam[0] / np.sqrt(variance)
        pd = self.prior_derivative - self.p @ self.prior_derivative
        # Re-standardizing the empirical prior changes the ridge in physical units.
        pd = pd - self.l2 * ((self.phi - geometry["prior_mean"])**2 - variance) * physical_lam
        influence = (q[:, None] * (hc - (self.phi - geometry["posterior_mean"]) @ response.T)
                     + self.p[:, None] * (pd @ response.T))
        return h, response, influence

    def predict(self, observables, *, labels, definition, absolute=True, validation_key=None):
        """Predict declared observables, including refit and normalization response."""
        if content_hash(self.configuration) != self.fit_configuration_sha256:
            raise ValueError("a fit input or feature definition has changed")
        if self.frozen_id and (array_identity(self.scale_weights) != self.freeze_record["weights"]
                               or array_identity(self.scale_lam) != self.freeze_record["multipliers"]):
            raise ValueError("a frozen fit weight or multiplier has changed")
        h, response, influence = self._response(observables)
        if h.shape[1] != len(labels):
            raise ValueError("one label is required per observable")
        validation_record = None
        if validation_key is not None:
            validation_record = plan_observable(h, labels=labels, definition=definition)
            if (not self.frozen_id or self.freeze_record["validation_plan"].get(validation_key)
                    != validation_record):
                raise ValueError("validation observables, labels or definition differ from the frozen plan")
            if not absolute:
                raise ValueError("validate raw absolute integrals before forming normalized ratios")
        mean = self.scale_weights @ h
        jac = response @ self.matrix @ self.data.normalization_jacobian
        if absolute:
            rate = self.data.mean[self.data.normalization]
            jac *= rate[0]
            jac[:, self.data.normalization] += mean[0]
            mean = mean * rate[:, None]
            influence *= rate[0]
        prior = {self.prior_key: dict(influence=influence, identity=self.prior_identity)}
        return Prediction(mean, tuple(labels), self.data.scale_names, _source(self.data, jac),
                          prior, dict(definition=definition, units="pb" if absolute else "unit area",
                                      configuration_sha256=content_hash(self.configuration),
                                      validation_key=validation_key, validation_record=validation_record),
                          (self.frozen_id,) if self.frozen_id else ())

    def diagnose(self, block, *, rank_cutoff=1e-6, significance=0.01):
        """Test supplied *unfitted* features using their full common-run covariance.

        Fitted columns are omitted by label; their closure is reported separately
        in ``central.report``. No supplied cross-covariance is ever replaced by
        an independence assumption. This diagnoses the supplied functions only.
        """
        if content_hash(self.configuration) != self.fit_configuration_sha256:
            raise ValueError("a fit input or feature definition has changed")
        _check_block(block, len(self.p), len(self.data.labels))
        for j, label in enumerate(block.labels):
            if label in self.labels:
                k = self.labels.index(label)
                if not all(np.array_equal(a, b) for a, b in (
                        (block.features[:, j], self.phi[:, k]),
                        (block.matrix[j], self.matrix[k]), (block.offset[j], self.offset[k]),
                        (block.prior_derivative[:, j], self.prior_derivative[:, k]))):
                    raise ValueError("a reused fitted label has a different feature or target definition")
        keep = [i for i, label in enumerate(block.labels) if label not in self.labels]
        labels = tuple(block.labels[i] for i in keep)
        h, response, influence = self._response(block.features[:, keep])
        matrix = block.matrix[keep]
        residual = self.scale_weights @ h - (block.offset[keep, None]
                                             + matrix @ self.data.normalized).T
        jac = (response @ self.matrix - matrix) @ self.data.normalization_jacobian
        dp = block.prior_derivative[:, keep]
        influence -= self.p[:, None] * (dp - self.p @ dp)
        prediction = Prediction(residual, labels, self.data.scale_names,
                                _source(self.data, jac),
                                {self.prior_key: dict(
                                    influence=influence, identity=self.prior_identity)},
                                dict(kind="training correlation diagnostic", block=block.identity))
        test = residual_test(residual, prediction.covariance, rank_cutoff=rank_cutoff,
                             significance=significance)
        return dict(block=block.name, labels=labels, residual=prediction, test=test,
                    omitted_fitted_labels=[x for x in block.labels if x in self.labels])

    def freeze(self, path, *, selection, validation_plan):
        """Save immutable choices before validation; refuse to overwrite a freeze.

        This records the declared chronology. It cannot establish whether a
        person previously inspected an external reference. Include all candidate
        outcomes and diagnostic thresholds in ``selection``, and exact cuts,
        bins and observable definitions in ``validation_plan``.
        """
        if not selection or not validation_plan:
            raise ValueError("record both the selection decision and the validation plan")
        for entry in validation_plan.values():
            if set(entry) != {"labels", "definition", "event_columns"}:
                raise ValueError("build each validation-plan entry with plan_observable")
        configuration = self.configuration
        if content_hash(configuration) != self.fit_configuration_sha256:
            raise ValueError("a fit input or feature definition has changed since fitting")
        modules = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        record = dict(configuration=configuration,
                      configuration_sha256=content_hash(configuration),
                      implementation_sha256=content_hash(modules), selection=json_value(selection),
                      validation_plan=json_value(validation_plan),
                      frozen_utc=datetime.now(timezone.utc).isoformat(),
                      weights=array_identity(self.scale_weights), multipliers=array_identity(self.scale_lam))
        record["freeze_id"] = content_hash(record)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, weights=self.scale_weights, lam=self.scale_lam,
                                manifest=np.array(canonical_json(record)))
        self.frozen_id, self.freeze_record = record["freeze_id"], record
        return record


def _check_block(block, n, d):
    if block.features.shape[0] != n or block.matrix.shape[1] != d:
        raise ValueError("block event order or numerical input dimension does not match")


def fit_constraints(prior_weights, data, blocks, *, prior_id, event_ids=None, l2=1e-4):
    """Fit arbitrary declared features with the existing maximum-entropy solver.

    The public ``upgrade`` interface remains convenient for standard towers.
    This lower-level interface adds a complete common-integral covariance for
    user-defined physical blocks. It does not select blocks automatically.
    """
    if data.role != "training":
        raise ValueError("only training inputs may determine a fit")
    p = np.asarray(prior_weights, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("prior weights must be finite and strictly positive")
    with np.errstate(over="ignore"):
        total = p.sum()
    if not np.isfinite(total):
        raise ValueError("the sum of prior weights must be finite")
    p = p / total
    if not prior_id or not np.isfinite(l2) or l2 < 0:
        raise ValueError("declare a prior_id and finite nonnegative ridge")
    ids = np.arange(len(p)) if event_ids is None else np.asarray(event_ids)
    if ids.shape != p.shape or len(np.unique(ids)) != len(p):
        raise ValueError("event IDs must be unique and match the weight order")
    blocks = tuple(blocks)
    for block in blocks:
        _check_block(block, len(p), len(data.labels))
    labels = tuple(label for block in blocks for label in block.labels)
    if len(set(labels)) != len(labels) or len({b.name for b in blocks}) != len(blocks):
        raise ValueError("fitted block names and feature labels must be unique")
    phi = np.column_stack([b.features for b in blocks]) if blocks else np.empty((len(p), 0))
    matrix = np.concatenate([b.matrix for b in blocks]) if blocks else np.empty((0, len(data.labels)))
    offset = np.concatenate([b.offset for b in blocks]) if blocks else np.empty(0)
    pd = np.column_stack([b.prior_derivative for b in blocks]) if blocks else np.empty_like(phi)
    if phi.shape[1] and (p @ (phi - p @ phi)**2 <= 1e-24).any():
        raise ValueError("constant/unsupported features must be removed before fitting")
    target = offset[:, None] + matrix @ data.normalized
    weights, lams = [], []
    for scale in range(data.mean.shape[1]):
        if phi.shape[1]:
            q, lam, ok = _newton_maxent(phi, p, target[:, scale], l2=l2,
                                        lam0=lams[0] if scale else None)
            if not ok:
                raise RuntimeError(f"constraint fit failed at coherent scale {data.scale_names[scale]}")
        else:
            q, lam = p.copy(), np.empty(0)
        weights.append(q)
        lams.append(lam)
    weights, lams = np.asarray(weights), np.asarray(lams)
    weights.setflags(write=False)
    lams.setflags(write=False)
    residual = weights @ phi - target.T
    target_cov = matrix @ data.normalized_covariance @ matrix.T
    error = np.sqrt(np.maximum(np.diag(target_cov), 0))
    pull = np.divide(np.abs(residual[0]), error, out=np.full(len(labels), np.nan), where=error > 0)
    stats = weight_diagnostics(weights[0], p)
    report = dict(lam=lams[0], mu=target[:, 0], feature_names=labels, Phi=phi, p=p,
                  constraint_residuals=residual, target_covariance=target_cov,
                  zero_error_labels=[label for label, e in zip(labels, error) if e == 0],
                  zero_error_residuals=residual[:, error == 0],
                  scale_names=data.scale_names, l2=l2,
                  uncertainty_note="input agreement, not independent prediction validation",
                  q_over_p_range=(float(np.min(weights[0] / p)), float(np.max(weights[0] / p))))
    relative = np.divide(np.abs(residual[0]), np.abs(target[:, 0]),
                         out=np.zeros(len(labels)), where=target[:, 0] != 0)
    unscaled = (error == 0) & (np.abs(residual[0]) > 1e-12 * (1 + np.abs(target[:, 0])))
    max_pull = float("nan") if unscaled.any() else float(np.max(pull[np.isfinite(pull)], initial=0))
    central = UpgradeResult(weights[0], 1 / (len(p) * (weights[0] @ weights[0])),
                            float(np.max(relative, initial=0)), float("nan"),
                            report, {}, {b.name: len(b.labels) for b in blocks},
                            effN_retained=stats["effective_fraction_retained"],
                            max_pull=max_pull)
    fit = ConstraintFit(central, weights, lams, data, blocks, p, ids, prior_id, l2,
                        phi, matrix, offset, pd, labels)
    fit.fit_configuration_sha256 = content_hash(fit.configuration)
    return fit


def load_frozen(path, prior_weights, data, blocks, *, prior_id, event_ids=None):
    """Restore saved weights after checking the exact reconstructed input/features.

    Feature arrays stay in the user's event representation; frozen archives do
    not duplicate them. No fit is performed when loading.
    """
    with np.load(path, allow_pickle=False) as archive:
        record = json.loads(str(archive["manifest"]))
        weights, lams = archive["weights"].copy(), archive["lam"].copy()
    freeze_id = record.pop("freeze_id")
    if content_hash(record) != freeze_id:
        raise ValueError("freeze manifest has changed")
    record["freeze_id"] = freeze_id
    # Construct a no-constraint shell; restore the declared basis without solving.
    fit = fit_constraints(prior_weights, data, [], prior_id=prior_id, event_ids=event_ids,
                          l2=record["configuration"]["l2"])
    fit.blocks = tuple(blocks)
    for block in fit.blocks:
        _check_block(block, len(fit.p), len(data.labels))
    fit.labels = tuple(label for block in fit.blocks for label in block.labels)
    if fit.blocks:
        fit.phi = np.column_stack([b.features for b in fit.blocks])
        fit.matrix = np.concatenate([b.matrix for b in fit.blocks])
        fit.offset = np.concatenate([b.offset for b in fit.blocks])
        fit.prior_derivative = np.column_stack([b.prior_derivative for b in fit.blocks])
    if (content_hash(fit.configuration) != record["configuration_sha256"]
            or array_identity(weights) != record["weights"]
            or array_identity(lams) != record["multipliers"]):
        raise ValueError("frozen event data, target input, features or stored fit do not match")
    fit.scale_weights, fit.scale_lam = weights, lams
    weights.setflags(write=False)
    lams.setflags(write=False)
    fit.central.weights = weights[0]
    fit.central.effN = 1 / (len(fit.p) * (weights[0] @ weights[0]))
    fit.central.effN_retained = (fit.p @ fit.p) / (weights[0] @ weights[0])
    fit.central.chosen_moments = {b.name: len(b.labels) for b in fit.blocks}
    target = fit.offset[:, None] + fit.matrix @ data.normalized
    residual = weights @ fit.phi - target.T
    target_cov = fit.matrix @ data.normalized_covariance @ fit.matrix.T
    error = np.sqrt(np.maximum(np.diag(target_cov), 0))
    pull = np.divide(np.abs(residual[0]), error, out=np.zeros(len(fit.labels)), where=error > 0)
    relative = np.divide(np.abs(residual[0]), np.abs(target[:, 0]),
                         out=np.zeros(len(fit.labels)), where=target[:, 0] != 0)
    fit.central.closure = float(np.max(relative, initial=0))
    unscaled = (error == 0) & (np.abs(residual[0]) > 1e-12 * (1 + np.abs(target[:, 0])))
    fit.central.max_pull = float("nan") if unscaled.any() else float(np.max(pull, initial=0))
    fit.central.report.update(lam=lams[0], Phi=fit.phi, feature_names=fit.labels,
                              mu=target[:, 0], constraint_residuals=residual,
                              target_covariance=target_cov,
                              zero_error_labels=[label for label, e in zip(fit.labels, error) if e == 0],
                              zero_error_residuals=residual[:, error == 0],
                              q_over_p_range=(float(np.min(weights[0] / fit.p)),
                                              float(np.max(weights[0] / fit.p))))
    fit.frozen_id, fit.freeze_record = freeze_id, record
    fit.fit_configuration_sha256 = record["configuration_sha256"]
    return fit


def residual_test(residuals, covariance, *, rank_cutoff=1e-6, significance=0.01,
                  discarded_tolerance=1e-8):
    """Report a correlated statistical discrepancy at each coherent scale.

    Chi-square tails are approximate numerical diagnostics. Discarded-mode
    residuals are reported explicitly and prevent a clean unresolved conclusion.
    A fully imposed block has no remaining test; it is never called validation.
    """
    r = np.asarray(residuals, float)
    if r.ndim == 1:
        r = r[None, :]
    if r.ndim != 2 or not np.isfinite(r).all():
        raise ValueError("residuals must be finite scale-by-observable values")
    if not 0 < rank_cutoff < 1 or not 0 < significance < 1:
        raise ValueError("rank cutoff and significance must be between zero and one")
    n = r.shape[1]
    cov = covariance_matrix(covariance, n)
    if not n:
        return dict(status="no_unfitted_constraints", rank=0, dimension=0,
                    resolved=None, p_values=None, rank_cutoff=rank_cutoff)
    diag = np.maximum(np.diag(cov), 0)
    positive = diag > 0
    sd = np.sqrt(np.where(positive, diag, 1))
    corr = cov / np.outer(sd, sd)
    vals, vec = np.linalg.eigh(corr)
    keep = vals > rank_cutoff * max(float(vals.max()), np.finfo(float).tiny)
    rank = int(keep.sum())
    scaled = r / sd
    projected = scaled @ vec
    discarded = np.linalg.norm(projected[:, ~keep], axis=1)
    zero_variance_residual = np.max(np.abs(r[:, ~positive]), axis=1, initial=0)
    incomplete = bool(np.any(discarded > discarded_tolerance))
    chi = ((projected[:, keep]**2) / vals[keep]).sum(axis=1)
    pvalues = _chi2_survival(chi, rank) if rank else None
    resolved = bool(np.max(pvalues) < significance) if rank else None
    status = ("resolved_at_all_scales" if resolved else "not_resolved_at_all_scales")
    if incomplete or not rank:
        status = "discarded_modes_require_review" if incomplete else "no_statistical_rank"
        if not resolved:
            resolved = None
    return json_value(dict(status=status, rank=rank, dimension=n, resolved=resolved,
                           chi2=chi, p_values=pvalues, eigenvalues=vals,
                           rank_cutoff=rank_cutoff, significance=significance,
                           discarded_standardized_norm=discarded,
                           zero_variance_residual=zero_variance_residual,
                           discarded_tolerance=discarded_tolerance,
                           criterion="all coherent scale residuals below the declared tail probability; central statistical covariance only"))


def _chi2_survival(chi, rank):
    t = np.asarray(chi, float) / 2
    if rank % 2:
        q, a = np.array([math.erfc(math.sqrt(x)) for x in t]), .5
    else:
        q, a = np.exp(-t), 1.
    while a < rank / 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.exp(a * np.log(t) - t - math.lgamma(a + 1))
        q += np.where(t == 0, 0, term)
        a += 1
    return np.clip(q, 0, 1)


def validate_prediction(prediction, reference, *, labels=None, independent_of, definition):
    """Compare a frozen prediction with declared independent raw reference integrals.

    ``independent_of`` must name every training sample used. Available run IDs
    are checked for overlap. This verifies the declared data separation, not an
    analyst's history or the adequacy of the chosen physical information.
    Use Prediction.transform for reference ratios after forming raw predictions.
    """
    if not prediction.frozen or reference.role != "validation":
        raise ValueError("independent validation requires a frozen fit and a validation input")
    if not prediction.provenance.get("validation_record"):
        raise ValueError("predict the frozen raw validation observable before forming transformations")
    if json_value(definition) != prediction.provenance["validation_record"]["definition"]:
        raise ValueError("reference definition differs from the frozen validation definition")
    samples = {v["identity"]["sample_id"] for v in prediction.sources.values()}
    if set(independent_of) != samples or reference.sample_id in samples:
        raise ValueError("explicitly declare a distinct reference independent of every training sample")
    for source in prediction.sources.values():
        if set(source["identity"]["run_ids"]) & set(reference.run_ids):
            raise ValueError("training and validation share run identifiers")
    if prediction.scale_names != reference.scale_names:
        raise ValueError("prediction and reference coherent scales must have identical ordering")
    names = tuple(labels or prediction.labels)
    matrix = reference.selector(names)
    target = Prediction((matrix @ reference.mean).T, prediction.labels, reference.scale_names,
                        _source(reference, matrix), {}, dict(definition=definition,
                                                             independent_of=sorted(samples)),
                        prediction.frozen)
    return dict(prediction=prediction, reference=target,
                difference=prediction.difference(target),
                definition=definition, independence="declared; available run IDs checked")


def plan_observable(values, *, labels, definition):
    """Fingerprint held-out event columns and their meaning before opening references."""
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != len(labels) or not np.isfinite(values).all():
        raise ValueError("planned observable columns must be finite and match their labels")
    return dict(labels=list(labels), definition=json_value(definition),
                event_columns=array_identity(values))
