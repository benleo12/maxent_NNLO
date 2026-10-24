# How to construct the moments (the fixed-order side)

The fixed-order input to `upgrade()` is a short list of numbers per constrained observable,
accumulated event by event in your fixed-order calculation. They are weighted sums, exactly
like a cross section, never a histogram. This is the whole fixed-order interface.

## 1. The moment

Map the observable v onto u ∈ [−1, 1] with the map that suits it:

    linear       (rapidities, angles):           u = 2 (v − lo)/(hi − lo) − 1
    logarithmic  (transverse momenta,
                  continuum masses):              u = 2 (ln v − ln lo)/(ln hi − ln lo) − 1
    Breit-Wigner (resonant masses):               u = 2 (t(v) − t(lo))/(t(hi) − t(lo)) − 1,
                                                  t(v) = arctan((v² − M²)/(M Γ))

Values outside [lo, hi] sit at the edge (clip u to [−1, 1]). Build the Chebyshev polynomials,

    T_0 = 1,   T_1 = u,   T_n = 2 u T_{n−1} − T_{n−2},

and accumulate, for n = 1 … N, over your phase-space points j with weights w_j inside the
fiducial region:

    μ_n = Σ_j w_j T_n(u(v_j)) / Σ_j w_j .

That is the same as booking the observable T_n(u(v)) as a one-bin histogram and dividing by
the fiducial cross section. The patch in `nnlojet_patch/` does exactly this for NNLOJET.

The map matters. It decides how much a few moments resolve:
- on the Drell-Yan mass, six moments on the Breit-Wigner map bring the spectrum to 0.9% of
  NNLO and twelve to 0.75%, while the same six on the linear map leave it at 1.1%, little
  better than the unconstrained prior (1.5%);
- the map must be identical on both sides, because the prior side evaluates the same T_n(u).
  On the Drell-Yan prior, declaring the linear map for moments booked on the Breit-Wigner map
  converges with clean closure and collapses N_eff/N from 99.9% to 10.8%.

## 2. Born towers and recoil windows

Every constrained observable is either a Born tower or a recoil window. The observable
decides which, not preference.

**Born tower.** The observable is fixed by the Born kinematics of the color singlet and is
infrared safe over its whole range: the mass, the rapidity, the Collins-Soper decay angle.
Its moments are imposed over the full fiducial range.

**Recoil window.** The observable vanishes at Born level, and its fixed-order prediction
diverges in the soft limit. Examples are the singlet transverse momentum, the acoplanarity
π − Δφ, and a_T.
- It is constrained only above the matching scale v_m, through the C² window
  w(v) = 6t⁵ − 15t⁴ + 10t³ with t = clip(ln(v/v_m)/ln 2, 0, 1). The window rises from v_m to
  2 v_m, and is cut at an upper edge c (500 GeV in the released tables, 1000 GeV for the Higgs
  and the jets).
- Accumulate the windowed moments and the window rate:

      μ_n = Σ_j w_j w(v_j) T_n(u(v_j)) / Σ_j w_j w(v_j),      r = Σ_j w_j w(v_j) / Σ_j w_j,

  with u the logarithmic map on [soft_lo, x_hi].
- That is all the fixed order supplies. `upgrade()` forms the blended target of each order,
  μ_n = (1 − r) ⟨T_n⟩_{p, 1−w} + r ⟨T_n⟩_{FO, w}, from these numbers and the prior itself
  (or, with `transition="windowed"`, imposes w T_n and w directly).
- The matching scale follows from power counting, α_s(p_T) ln²(Q/p_T) < 0.2 with Q the hard
  scale (`maxent_upgrade.matching_scale(Q)`): 27.9 GeV for Q = m_Z, 24.9 GeV for Q = m_W,
  27.6 GeV for diphoton production (Q = 90 GeV) and 37.1 GeV for Q = m_H. The released tables
  use 30 (rounded up), 25, 28 and 37 GeV.

**Joint moments.** To constrain a correlation that the marginal towers cannot reach, book the
products ⟨w_x w_y T_m(u(x)) T_n(u(y))⟩/⟨w_x w_y⟩ for m, n = 1 … 4 and their rate
⟨w_x w_y⟩/⟨1⟩, with w = 1 for an observable that has no window. They are imposed in the
windowed form, and every recoil tower they couple must be windowed too.

**Matching uncertainty.** To estimate it, book the same windowed moments and rate with a few
other windows, for instance rising over [⅔ v_m, ⁴⁄₃ v_m], [⁴⁄₃ v_m, ⁸⁄₃ v_m] and [v_m, 4 v_m],
and re-solve with each (`load_moments(process, window=V)` for the released Higgs and diphoton
variants). Dedicated runs need only the channels with a nonzero recoil and can reuse the main
run's Vegas grids; their rate then needs σ_fid from all channels of the main run.

Putting a recoil observable in the Born slot is not merely inaccurate. Imposing the
full-range moments of π − Δφ contradicts the window, which keeps the shower below the
matching scale, and the solve has no positive-weight solution. A convergence failure of
this kind is a diagnosis, not a numerical problem to tune away.

## 3. Errors

Return two errors per moment:
- the **Monte Carlo** error: the scatter of independent runs divided by √(number of runs);
- the **scale** error: the half-range of μ_n over the seven-point variation.

Pool the runs as the ratio of summed numerators to summed denominators, not as the mean of
per-run ratios. `upgrade()` uses the two errors combined in quadrature (`errors`) to decide
which orders the input resolves. Keep the per-run, per-scale accumulators: they are what the
uncertainty bands re-pool (see the `_runs.json.gz` files in `moments/`).

## 4. How many moments

Return moments up to a generous depth (12 for Born towers, 20 for a recoil window is ample).
`upgrade()` imposes each tower up to the highest order whose shift the input resolves,

    SNR_n = |μ_n(fixed order) − μ_n(prior)| / σ_n > 1,

and drops the orders beyond it. This is why σ_n matters as much as μ_n.

## 5. Which calculation each moment comes from

Label each tower by the fixed-order calculation it comes from, not by the inclusive order of
the process. From an inclusive color-singlet calculation at NNLO:
- the singlet mass and rapidity, inclusive over radiation, are NNLO;
- the singlet transverse momentum is not NNLO. A nonzero recoil needs a real emission, so
  the inclusive NNLO result gives the recoil at NLO(V+jet).

To constrain the recoil at NNLO, take its window from the V+jet calculation at NNLO.

## 6. Practical path

1. Fix the observable list, the maps and the windows at low statistics.
2. Raise statistics for production. Integrated moments converge long before a differential
   spectrum would be smooth.
3. Send back, per tower and order, μ_n with its two errors, and the window rate for each
   recoil.

`export_moments.py` shows how the released tables were assembled from NNLOJET runs.
`maxent_upgrade.compute_fo_moments` does the same, joint blocks included, from fixed-order
events held in Python.
