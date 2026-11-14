# Independent W prediction reference

`V_WP.npz` and `V_WM.npz` retain the final independent NNLOJET common-vector
integrals, complete central covariance and per-seed accumulators. These are
validation inputs; the training loader refuses to use them for selecting
constraints. All 10,372 physical runs are retained, with 250,000 integration
points per run and 2.593 billion points in total.

| Charge | LO | R | V | RR | RV | VV |
|---|---:|---:|---:|---:|---:|---:|
| W+ | 256 | 2048 | 128 | 1899 | 1024 | 64 |
| W− | 256 | 2048 | 128 | 1871 | 585 | 65 |

`mean` is the 112-integral vector at seven coherent scales, in pb. `cov` is the
complete central covariance of those raw estimates, in pb². `acc`, `seeds` and
`n_runs` retain unequal channel counts: unused rows have seed `-1` and NaN
accumulators. Use actual rows independently within each additive channel;
padding is not an observation. The rate entry is the same vector's baseline
normalization, so derived fractions and charge asymmetries retain denominator
and cross-observable covariance.

The ordered `(muR,muF)` factors are `(1,1)`, `(2,2)`, `(0.5,0.5)`, `(2,1)`,
`(1,2)`, `(0.5,1)`, `(1,0.5)`, relative to the charged-lepton–neutrino invariant
mass. Every actual card and its scale order was checked. Scale columns are
coherent alternatives, not seven statistical observations. Training and
reference physical seeds are disjoint; the accompanying audits record their
separate production and frozen prediction definitions.

The predeclared finite stopping rule ended after two precision-only extensions.
Every scalar acceptance precision target passes. All eleven 28 GeV and all
eleven transverse-mass-selection asymmetry bins pass the 0.002 target. Three
32 GeV asymmetry bins—indices 0, 3 and 4—have errors 0.002136, 0.002042 and
0.002164, slightly above that target. These bins and every original seed remain
in the results. No model or binning was changed after this audit.

The first 32 GeV asymmetry bin is still sensitive to valid W+ RV seed 46233,
which contributes about 70.6% of its estimated variance. Numerical standard
errors in this bin should not be interpreted as calibrated Gaussian tail
probabilities or used for a high-significance claim. `LARGE_VARIANCE_AUDIT.json`
retains the detailed checks. `PRECISION_ONLY.json` and
`PRECISION_STOPPING_COMPLETE.json` record every target and the final decision.

SHA256:

- W+: `32259faba4df462953a8bdd047c3d59acec76e191dd1f01706b38bca51eed45c`
- W−: `e7afc5fcb45a93e7fd02f85c3d61e54f7f8a40ac4c4360f7fcfec582528555a4`
