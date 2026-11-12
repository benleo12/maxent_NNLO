"""The weights and diagnostics returned by a reweighting."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class UpgradeResult:
    """What `upgrade()` returns.

    Attributes
    ----------
    weights : np.ndarray
        The new per-event weights q_i, strictly positive, summing to one.
    effN : float
        Kish effective sample fraction 1/(N sum q_i^2), in (0, 1].
    effN_retained : float
        Fraction of the PRIOR's own Kish count that the upgrade keeps,
        Kish(q)/Kish(p).  Equals effN for an unweighted prior; for a weighted (e.g.
        sliced) prior it is the meaningful cost of the upgrade.
    closure : float
        Worst relative deviation of an achieved moment from its target, max|ach/mu - 1|,
        over the imposed moments (Born, window and joint).  Near-zero targets inflate it;
        see max_pull.
    max_pull : float
        Worst |achieved - target| / sigma over the imposed moments, with sigma the target's
        Monte Carlo (stat) error where the moments carry `stat_errors`, otherwise its total
        error.  The window shares (the rate of a windowed tower or joint block, the share
        above a blended tower's hard cut) are constraints too, but not moments: their
        closure and pull are in report['shares'] and printed by describe().
    x_match : float
        Matching scale of the first recoil window (nan without one).
    moment_snr : dict {observable: array}
        Resolution spectrum SNR_n = |mu_n(target) - mu_n(prior)| / sigma_n, orders 1..N.
    chosen_moments : dict {observable or joint key: int}
        Number of moments imposed per tower (towers are imposed up to their highest
        resolved order) and per joint block.
    report : dict
        Multipliers `lam`, targets `mu`, `feature_names`, `towers` (per-tower table used by
        describe(); `kind` is 'born', 'blend', 'windowed' or 'joint'), `moment_selection`,
        `windows` ({recoil observable: x_match, x_hi, rate, transition, profile}),
        `shares` ({constraint: {closure, pull}} for the window shares), `q_over_p_range`,
        the soft-constraint block, and with `keep_features=True` the feature matrix `Phi`
        and prior `p`.
    band : None
        Reserved; bands are re-solves of upgrade() (see README, "Uncertainty bands").
    soft_pulls : np.ndarray or None
        soft_constraints=True only: (achieved - target)/sigma per constraint.
    """
    weights: np.ndarray                 # new positive per-event weights (central solve)
    effN: float                         # effective sample FRACTION in (0, 1]
    closure: float                      # worst relative moment closure |achieved/target - 1|
    x_match: float                      # matching scale of the first recoil window
    report: Dict[str, Any]              # multipliers, targets, per-tower table, moment selection
    moment_snr: Dict[str, np.ndarray]   # per-observable SNR spectrum (orders 1..Nmax)
    chosen_moments: Dict[str, int] = field(default_factory=dict)  # #moments imposed per tower
    band: Optional[Dict[str, np.ndarray]] = None   # legacy histogram path only
    # soft mode: (achieved - target)/sigma per constraint
    soft_pulls: Optional[np.ndarray] = None
    effN_retained: float = float("nan")            # Kish(q)/Kish(p)
    max_pull: float = float("nan")                 # worst |achieved - target|/sigma

    def summary(self) -> str:
        """One line with the retained statistics and moment agreement."""
        retained = (f"retained statistics {100 * self.effN_retained:.2f}%"
                    if np.isfinite(self.effN_retained)
                    else f"effective fraction {100 * self.effN:.2f}%")
        residual = (f"largest moment residual {self.max_pull:.3g} MC errors"
                    if np.isfinite(self.max_pull) else f"relative moment residual {self.closure:.2g}")
        return f"{len(self.weights):,} events | {retained} | {residual} | positive weights: {bool((self.weights > 0).all())}"

    def describe(self) -> str:
        """Readable diagnostics; detailed numerical arrays remain in report."""
        towers = self.report.get("towers") or []
        width = max([len(t["name"]) for t in towers] + [10])
        out = ["MaxEnt reweighting", self.summary(), "",
               f"{'Observable':<{width}}  {'Constraint':<10}  {'Map':<4}  {'Moments':>9}  {'Residual / MC error':>19}"]
        for tower in towers:
            pull = tower.get("max_pull", float("nan"))
            residual = f"{pull:.3g}" if np.isfinite(pull) else "not supplied"
            out.append(f"{tower['name']:<{width}}  {tower['kind']:<10}  {str(tower.get('map', '-')):<4}  "
                       f"{tower['imposed']:>3} / {tower['supplied']:<3}  {residual:>19}")
        if towers:
            out.append(
                "Moments: imposed / available. Residuals use the input Monte Carlo errors when supplied.")
        shares = self.report.get("shares") or {}
        if shares:
            out.extend(["", "Additional rate constraints:"])
            for name, values in shares.items():
                pull = values['pull']
                residual = f"{pull:.3g} MC errors" if np.isfinite(pull) else "MC error not supplied"
                out.append(f"  {name}: {values['closure']:.2g} relative residual, {residual}")
        out.extend(["", "Weights sum to one. Multiply by the target cross section for absolute predictions.",
                    f"Effective events: {len(self.weights) * self.effN:,.1f}",
                    f"Largest relative moment residual: {self.closure:.3g}"])
        if "q_over_p_range" in self.report:
            lo, hi = self.report["q_over_p_range"]
            out.append(f"Weight ratio q/p: {lo:.3g} to {hi:.3g}")
        return "\n".join(out)
