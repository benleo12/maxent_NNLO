"""Check hard targets against the convex hull of a finite event sample."""
import numpy as np


class InfeasibleTarget(ValueError):
    """A measured direction separates the target from every prior event."""


def _closest_point(vertices, weights):
    """Project the origin onto an active simplex, dropping negative coefficients."""
    # The current weights are feasible. Move towards the unconstrained affine
    # minimum until reaching it or until a coefficient vanishes.
    while len(weights) > 1:
        axes = (vertices[:-1] - vertices[-1]).T
        first = np.linalg.lstsq(axes, -vertices[-1], rcond=None)[0]
        affine = np.r_[first, 1.0 - first.sum()]
        if np.min(affine) >= 0:
            return vertices, affine
        neg = affine < 0
        ratios = np.where(neg, weights / np.where(neg, weights - affine, 1.0), np.inf)
        edge = np.argmin(ratios)
        weights = (1 - ratios[edge]) * weights + ratios[edge] * affine
        weights[edge] = 0.0
        keep = weights > 0
        weights, vertices = weights[keep], vertices[keep]
        weights /= weights.sum()
    return vertices, weights


def verify_support(features, target, prior_mean, names):
    """Return the convex-hull status, or reject with a verified separating direction.

    The nearest-point active set uses only small linear systems. A subset of rows
    can prove that a target is attainable. Before rejecting any target we check
    the separating bound on every event. An inconclusive search never rejects it.
    """
    n, k = features.shape
    if not k:
        return "no hard shape constraints"
    low, high = features.min(axis=0), features.max(axis=0)
    scale = np.maximum(1.0, np.maximum(abs(low), abs(high)))
    eps = 64 * np.finfo(float).eps * (k + 1)
    outside = (target < low - eps * scale) | (target > high + eps * scale)
    if outside.any():
        j = np.flatnonzero(outside)[0]
        raise InfeasibleTarget(f"hard target {names[j]} = {target[j]:.8g} is outside prior support "
                               f"[{low[j]:.8g}, {high[j]:.8g}]")
    endpoint = ((target == low) | (target == high)) & ((high - low) > eps * scale)
    if endpoint.any():
        j = np.flatnonzero(endpoint)[0]
        raise InfeasibleTarget(f"hard target {names[j]} = {target[j]:.8g} equals a support endpoint "
                               "with events away from it; attaining it exactly requires zero event weights")
    if k == 1:
        return "in convex hull to numerical precision"
    width = np.maximum(high - low, scale * eps)
    # The prior mean itself is in the hull. Add only those actual events
    # needed to express the target, starting with a small deterministic subset.
    vertices = np.array([(prior_mean - target) / width])
    coefficients = np.ones(1)
    sample = features[np.linspace(0, n - 1, min(n, max(64, 4 * k)), dtype=int)]

    def full_bound(direction):
        scores = features @ direction
        index = int(np.argmax(scores))
        largest, goal = scores[index], target @ direction
        if goal > largest + eps * (scale @ abs(direction)):
            order = np.argsort(abs(direction))[::-1]
            terms = [f"{direction[j]:+.7g}*{names[j]}" for j in order if abs(direction[j]) > 1e-12]
            raise InfeasibleTarget("incompatible hard targets on this prior: projection "
                                   + " ".join(terms) + f" has target {goal:.9g}, above the "
                                   f"largest value on any input event ({largest:.9g})")
        return features[index]

    # Exact affine relations between observables survive any row selection. The
    # small matrix supplies candidate directions; the full sample must confirm
    # a contradiction. This also resolves near-roundoff differences that a
    # closest-point search on a nearly flat simplex cannot distinguish.
    centered = (sample - sample[0]) / width
    _, singular, axes = np.linalg.svd(centered, full_matrices=len(sample) < k)
    rank = np.count_nonzero(singular > eps * max(1., singular[0]))
    null = axes[rank:]
    if len(null):
        offset = null @ ((target - sample[0]) / width)
        if np.max(abs(offset)) > eps:
            direction = (offset @ null) / width
            full_bound(direction / np.max(abs(direction)))
            if len(sample) < n and len(null) > 1:
                # Rare coordinates may vanish on the subset. Identify which
                # combinations remain dependent on all rows, reducing only a
                # nullity-by-nullity Gram matrix in bounded batches.
                gram = np.zeros((len(null), len(null)))
                for start in range(0, n, 8192):
                    part = (features[start:start + 8192] - sample[0]) @ (null / width).T
                    gram += part.T @ part
                eigen, vectors = np.linalg.eigh(gram)
                exact = vectors[:, eigen <= eps * max(1., eigen[-1])]
                if exact.size:
                    projected = exact @ (exact.T @ offset)
                    direction = (projected @ null) / width
                    if np.max(abs(direction)) > eps:
                        full_bound(direction / np.max(abs(direction)))
    for _ in range(8 * (k + 1) + 32):
        nearest = coefficients @ vertices
        if (abs(nearest * width) <= eps * scale).all():
            return "in convex hull to numerical precision"
        direction = -nearest / width
        direction /= np.max(np.abs(direction))
        tolerance = eps * (scale @ abs(direction))
        goal = target @ direction
        small = sample @ direction
        row = sample[np.argmax(small)]
        # If the subset has no useful descent, query every input event.
        if small.max() <= goal + tolerance:
            row = full_bound(direction)
        candidate = (row - target) / width
        if np.any(np.all(vertices == candidate, axis=1)):
            break
        vertices, coefficients = _closest_point(np.vstack((vertices, candidate)), np.r_[coefficients, 0.0])
    return "convex-hull membership unresolved"
