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
| `EvalFuncs.f90` | `cheb_T(u,n)` (clipped Chebyshev recurrence), `prof_w(x,a,b,c,d)` (C² smootherstep matching window), and the `chebT_*` / `chebTw_*` observables for DY, Z+jet, diphoton and gg→H. |
| `Observables.f90` | `bind_obs` registrations, **each inside its own process `case` block** (`Z`/`ZJ`, `H`, `GG`). |

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
  `p(:,npar-1)+p(:,npar)` as for a colour-singlet `V`. Bind with `min_nphotons=2`.
* **Bind `eval_mem` observables with negative `imem`.** The dispatch uses
  `abs(imem)`, while `imem > 0` triggers a memoisation-reset call `eval_mem(0,0)`
  that a non-memoised eval must not receive.
* **Runcards need a `warmup`**, otherwise production aborts with a missing VEGAS
  grid file.
* **Pool seeds, don't average ratios**, and only pool seeds for which *every*
  channel is present — a missing Born channel silently wrecks the denominator.
