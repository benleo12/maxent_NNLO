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

The map controls how efficiently a finite set of moments represents the correction.
It must be identical on the prior and fixed-order sides. Declaring another map
without recomputing the target moments imposes a different problem, even if the
solver reports good moment agreement.

## 2. Born towers and recoil windows

Choose the constraint according to the region in which fixed order reliably
describes the observable. Born variables can also need a restricted window near
an acceptance edge, as in the W decay-angle example.

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
- The matching scale follows from power counting, α_s(p_T) (C_i/C_F) ln²(Q/p_T) < 0.2 with Q the
  hard scale and C_i/C_F the color factor of the incoming partons relative to quarks
  (`maxent_upgrade.matching_scale(Q, color=...)`): 28.0 GeV for Q = m_Z, 25.0 GeV for Q = m_W,
  27.7 GeV for diphoton production (Q = 90 GeV) and, with C_A/C_F = 9/4, 54.3 GeV for gluon
  fusion at Q = m_H (37.3 GeV without the color factor). The released tables use 30 (rounded
  up), 25, 28 and 54 GeV. The main Higgs table books 37 GeV, the letter's window.

**Joint moments.** To constrain a correlation that the marginal towers cannot reach, book the
products ⟨w_x w_y T_m(u(x)) T_n(u(y))⟩/⟨w_x w_y⟩ for m, n = 1 … 4 and their rate
⟨w_x w_y⟩/⟨1⟩, with w = 1 for an observable that has no window. They are imposed in the
windowed form, and every recoil tower they couple must be windowed too.

**Matching uncertainty.** The threshold 0.2 of the rule is a choice. Varying it by a factor of
two moves the transition region down to about [⅔ v_m, ⁴⁄₃ v_m] (expansion parameter about 0.4)
and up to about [⁴⁄₃ v_m, ⁸⁄₃ v_m] (about 0.1). To estimate the uncertainty, book the same
windowed moments and rate with these two windows and re-solve with each
(`load_moments(process, window="down")` and `window="up"` for the released tables).
Dedicated runs need only the channels with a nonzero recoil and can reuse the main run's Vegas
grids; their rate then needs σ_fid from all channels of the main run. If they also book the
nominal window, a moved window can enter as its difference to the nominal one, added to the main
run's moments; the difference needs far fewer points than the moments themselves. A window may
start below the lower end of the map: u is clipped to [−1, 1] there, in the fixed-order code
and in the package alike.

A bounded measurement function alone does not make fixed order reliable near a
Born singularity. Choose the window in the fixed-order calculation and use that
same definition on the prior. A numerical failure can indicate incompatible
moments, insufficient support or inconsistent definitions, and needs inspection.

## 3. Errors

Return two errors per moment:
- the **Monte Carlo** error: the scatter of independent runs divided by √(number of runs);
- the **scale** error: the half-range of μ_n over the seven-point variation.

Pool the moments over the runs as the ratio of summed numerators to summed denominators, not
as the mean of per-run ratios: one run with a small denominator would otherwise throw a wild
value into the average. The released tables take each window rate, whose per-run values are
all well behaved, as the mean of the per-run rates, the convention of the authors' solves; the
ratio of sums agrees with it within the run-to-run error. `upgrade()` uses the two errors
combined in quadrature (`errors`) to decide which orders the input resolves. Keep the per-run, per-scale accumulators: they are what the
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
