# How to construct the moments (recipe for the fixed-order side)

You provide, per constrained observable, a short list of numbers computed **event
level** in your fixed-order calculation. Nothing is binned. This is the entire
fixed-order interface to the reweighting.

## 1. The moment

For an observable x with a chosen range [lo, hi] and a map (linear or logarithmic),
map x to the interval [-1, 1]

    linear (mass, rapidity):   u(x) = 2 (x - lo)/(hi - lo) - 1
    logarithmic (transverse):  u(x) = 2 (ln x - ln lo)/(ln hi - ln lo) - 1

build the Chebyshev polynomials by the standard recurrence

    T_0(u) = 1,   T_1(u) = u,   T_n(u) = 2 u T_{n-1}(u) - T_{n-2}(u),

and return, for n = 1, 2, ..., N, the single number

    mu_n = ( sum_j w_j T_n(u(x_j)) ) / ( sum_j w_j )

where the sum runs over your fixed-order phase-space points j with weight w_j, at the
target order N+k, inside the fiducial volume. Also return the Monte-Carlo uncertainty
sigma_n on each mu_n, computed the same way you get the error on any booked observable
(seed or replica spread, and the seven-point scale envelope). The pair (mu_n, sigma_n)
per order n is the target. This is identical to booking the observable T_n(u(x)); there
is no histogram.

## 2. Which slot: Born or recoil

Every observable goes in exactly one of two slots, and the choice is dictated by the
observable, not by preference.

**Born slot** — the observable is defined at Born level and is infrared-safe over its
whole range. Its fixed-order moments may be imposed everywhere. Examples: the invariant
mass, the rapidity, the Collins-Soper `|cos theta*|`.

**Recoil slot** — the observable vanishes identically at Born level and its fixed-order
prediction diverges in the soft limit. It may only be constrained through the profile,
above the matching scale, with the shower left alone below. Examples: `pT` of the
colour-singlet system, `pi - dphi`, `a_T`.

Putting a recoil observable in the Born slot is not merely inaccurate — it makes the
dual **infeasible**. Imposing the fixed-order moments of `pi - dphi_gg` over its full
range directly contradicts the recoil profile, which deliberately preserves the shower
below the seam, and the MaxEnt solve has no positive-weight solution at any moment order
(verified for N = 3, 4, 6; `eventlevel/aa_angular_test.py`). `|cos theta*|`, a genuine
Born observable, converges at all of them. A convergence failure of this kind is a
diagnosis, not a numerical problem to be tuned away.

Constraining the Born angular observable matters. For the ATLAS 8 TeV diphoton fiducial
region the photon `pT` cuts lock `|cos theta*|` to `m_gg`, so mass moments alone move
the angle the wrong way. Against ATLAS 1704.03839 (median `|shape/data - 1|`):

| observable | prior | Born {m} | Born {m, cos} |
|---|---|---|---|
| `m_gg`       | 48.4% | 12.0% *(c)* | **9.5%** *(c)* |
| `pT_gg`      | 84.1% | 14.0% *(c)* | 13.9% *(c)* |
| `|cos theta*|` | 4.0% | 17.8% | **6.3%** *(c)* |
| `dphi_gg`    | 31.8% | 15.5% | **8.4%** |
| `a_T`        | 57.6% | 13.2% | 15.1% |

*(c)* = constrained. Adding the angle recovers it *and* relieves the mass, because the
two are correlated by the fiducial cuts. `dphi_gg` is never constrained in either column
and improves anyway. Note also that the prior's 4.0% on `|cos theta*|` was an accidental
cancellation; after the upgrade the value is pinned to the fixed order rather than
landing near the data by luck.

## 3. Range and map per observable

Use the same [lo, hi] and map you tell the reweighting code (they must match). Mass and
rapidity use the linear map over the fiducial range. Transverse momenta use the
logarithmic map. For a recoil constrained only above a resolution scale x_match, compute
the moment over [x_match, x_hi] with u mapped on [soft_lo, x_hi] (log), and also return
the fixed-order cross-section fraction in [x_match, x_hi] (the window rate).

## 4. How many moments

Return moments up to a generous ceiling (say N = 12). The reweighting keeps moment n
only while it is resolved above its own fixed-order uncertainty,

    SNR_n = | mu_n(order N+k) - mu_n(prior) | / sigma_n  >  1,

and drops the rest. This is why sigma_n matters as much as mu_n. A moment the fixed-order
calculation does not determine above its scale-and-statistics error is not imposed.

## 5. The orders, stated precisely (this is where we must be careful)

For an inclusive color-singlet calculation at NNLO:

  * the singlet MASS and RAPIDITY, inclusive over radiation, are genuinely NNLO;
  * the singlet TRANSVERSE MOMENTUM (recoil) is NOT NNLO from the inclusive calculation.
    pT > 0 requires a real emission, so the inclusive-V NNLO result gives the recoil
    spectrum at NLO(V+jet) accuracy. To constrain the recoil at NNLO you must take it
    from the V+jet calculation at NNLO, above a jet-resolution cut.

So label each constraint by the fixed-order calculation it comes from, not by the
inclusive order of the process:

  DY, inclusive Z at NNLO      -> m(ll), y(ll)                 constrained at NNLO
  DY, Z+jet at NNLO            -> pT(ll)/pT(j1) above cut      constrained at NNLO
  DY, Z+jet at NLO             -> pT(j2), dphi(l1,l2) off pi   constrained at NLO
  below the resolution cut     -> parton shower preserved      (its own constraint)

The diphoton case is the same with gamma gamma in place of Z. m(AA), y(AA), and the
photon rapidities come from inclusive gamma gamma at NNLO; pT(AA)/pT(j1), the
gamma gamma + jet masses, dY(A1,j1), and dphi(A1,A2) away from pi come from
gamma gamma + jet at NNLO above a resolution cut; pT(j2) is NLO. The individual photon
transverse momenta pT(A1), pT(A2) are NNLO for the inclusive observable, but their
features that depend on the recoil inherit the NLO(V+jet) accuracy of the recoil, so
state them as inclusive-NNLO with an NLO recoil tail rather than uniformly NNLO.

## 6. Practical path

Start at low statistics to fix the observable list and the maps, then raise statistics
for production. The moments converge fast because they are integrated quantities, so the
statistical precision on mu_n is reached long before a differential spectrum would be
smooth. Send back, per observable and per order, the arrays mu_n and sigma_n (and the
window rate for the recoil). That is all the reweighting needs.
