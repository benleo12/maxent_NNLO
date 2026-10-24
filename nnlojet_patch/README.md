# NNLOJET patch: event-level Chebyshev moments

These four patches teach NNLOJET to accumulate **event-level moments** — the
literal per-phase-space-point weighted sum

    <T_n> = sum_j w_j T_n(u(x_j)) / sum_j w_j

— rather than a moment reconstructed from a binned distribution. Everything the
reweighting consumes as a fixed-order target is produced this way.

Verified against NNLOJET v1.0.2 (`driver/core/`): each patch applies to the
pristine source and reproduces the working tree byte for byte.

```bash
cd <nnlojet>/driver/core
for f in Histograms IOHelper EvalFuncs Observables; do
  patch -p0 "$f.f90" < <this-dir>/"$f.f90.patch"
done
cd <nnlojet>/build && make -j6 NNLOJET
```

**On macOS, re-sign after installing.** Copying a freshly linked arm64 binary
invalidates its ad-hoc signature and the copy then hangs at exec with 0% CPU and
no output (it blocks on the security daemon, which looks exactly like a deadlock):

```bash
cp build/NNLOJET install/bin/NNLOJET
cp build/libnnlojet_core.dylib install/lib/
codesign -f -s - install/bin/NNLOJET
codesign -f -s - install/lib/libnnlojet_core.dylib
```

## What each patch does

| file | change |
|---|---|
| `Histograms.f90` | adds a `profile` flag to `Histogram_t`; a profile histogram accumulates `w * valObs` instead of `w`. Threaded through `newVariableWidth` / `newEqualWidth`. |
| `IOHelper.f90` | parses `profile = T` on a histogram line and passes it to `register_hist`. |
| `EvalFuncs.f90` | `cheb_T(u,n)` (clipped Chebyshev recurrence), `prof_w(x,a,b,c,d)` (C² smootherstep matching window), and the `chebT_*` / `chebTw_*` observables for DY, Z+jet, diphoton, gg→H and W± (`m_W` Breit-Wigner tower, `|y_W|`, profiled `pT_W`; the signed Collins-Soper decay angle `costh_w` with a full-range tower `chebT_ctw_n`, a windowed tower `chebTw_ctw_n` (flat for |cos θ*| < 0.6, C² ramp to zero at 0.7; see the errata below) and joint `chebTT_ptwctw_mn` moments with the recoil window). |
| `Observables.f90` | `bind_obs` registrations, **each inside its own process `case` block** (`Z`/`ZJ`, `H`, `GG`, `WM`, `WP`; the W blocks differ only in which of the last two particles is the charged lepton). |

## Usage

A moment is a single bin, so book it with `nbins=1` and a companion normal
histogram for the denominator; the bin width cancels in the ratio.

```
  chebT_mll_1 > norm_born   min=-1.001 max=1.001 nbins=1
  chebT_mll_2 > prof_mll_2  min=-1.001 max=1.001 nbins=1 profile = T
```

giving `<T_2(m_ll)> = prof_mll_2 / norm_born`. For a windowed recoil the profile
`w(pT)` is folded into the eval itself (`chebTw_*`), so

    <T_n>_w = prof_wptz_n / prof_wptz_0 ,   R = prof_wptz_0 / norm_born .

## Pitfalls found the hard way

* **Register in the right `case` block.** `init_obs()` branches on the process;
  bindings placed in the `Z` block are invisible to `H` and `GG`
  (`getIdFromName_obs: unmatched observable`). Note the `H` case statement spans
  continuation lines — insert after the *complete* statement.
* **Photons are `kin(npar)%pphotons(:,1..2)`** (pT-ordered) in `GG`, *not*
  `p(:,npar-1)+p(:,npar)` as for a color-singlet `V`. Bind with `min_nphotons=2`.
* **Bind `eval_mem` observables with negative `imem`.** The dispatch uses
  `abs(imem)`, while `imem > 0` triggers a memoisation-reset call `eval_mem(0,0)`
  that a non-memoised eval must not receive.
* **Runcards need a `warmup`**, otherwise production aborts with a missing VEGAS
  grid file.
* **Pool seeds, don't average ratios**, and only pool seeds for which *every*
  channel is present — a missing Born channel silently wrecks the denominator.

## Errata in the source comments

The patches reproduce the production source byte for byte (re-checked on 2026-10-01: all
four reverse-apply cleanly to the tree that produced the released moments), so two stale
comments are left as they were run:

* The comment above `eval_chebT_mll` says "linear map on [66,116] GeV".  The function uses
  the Breit-Wigner map `bw_u` with the Z mass and width, and the released Drell-Yan and Z+jet
  tables declare `map: bw` accordingly.
* The comments above `eval_w_ctw`, its `bind_obs` description and the comments in
  `runcards/w_moments_W{P,M}.run` say "|c| < 0.5, ramp to 0.7".  The compiled parameters
  (pa, pb, pc, pd) = (0.3, 0.4, 1.6, 1.7) act on cos θ* + 1, so the window is 1 for
  |cos θ*| ≤ 0.6 and 0 for |cos θ*| ≥ 0.7.

* The comment block above `eval_chebT_yh` gives the Higgs window as "[30,60]->1000"; the
  compiled `eval_chebTw_pth` and `eval_w_pth` use (37, 74, 1000), as the released table states.
  The `bind_obs` descriptions of the m_ll towers say "lin 66-116"; the functions use the
  Breit-Wigner map (above).
* The DY decay-angle accumulators (`chebT_ctl_n`, `chebTT_mllct_mn`) book the ABSOLUTE
  Collins-Soper angle, T_n(2|cos θ*| − 1) on [0, 1] (`eval_abscosthstCS_ll`); the released
  tables call the variable `abs_costh`.

The `chebW_*` observables (weighted polynomials T_n(u)/sqrt(1−u²)) belong to a basis test;
no run card books them and no table uses them.

## Run cards (`../runcards/`)

One file per production behind a released table or a figure of the papers, as run.  Each
production ran one perturbative channel per job; the CHANNELS block of the shipped card shows
one of them (LO, or R for `dy_ptz_full.run`), the others differ only in that line.

| run card | production | used for |
|---|---|---|
| `dy_moments_log30_hi.run` | Z at NNLO, 298 runs, μ = m_ll | `moments/dy_nnlo_moments` |
| `w_moments_WP.run`, `w_moments_WM.run` | W± at NNLO, 40 runs each, μ = m_ℓν | `moments/w_nnlo_moments_W{P,M}` |
| `gg_moments2.run` | γγ at NNLO, 40 runs, μ = m_γγ | `moments/aa_nnlo_moments` |
| `ggh_moments2.run` | H at NNLO (HEFT), 40 runs, μ = m_H; also the follower histograms of the leading jet | `moments/ggh_nnlo_moments`; the fixed-order jet reference |
| `zj_moments.run` | Z+jet at NLO, 40 runs, μ = H_T/2 | `moments/zj_nlo_moments` |
| `dy_ptz_full.run` | the real channels (R, RR, RV) of the Z run with a full-range p_T^ℓℓ histogram, 6 runs | the fixed-order recoil curve below the matching scale in the matching figure |
