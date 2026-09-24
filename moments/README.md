# Fixed-order moments, as used in the papers

`dy_nnlo_moments.csv` / `.json` — the NNLO Drell-Yan moments behind every
Drell-Yan number in both papers.

Process `pp -> Z -> l+l-` at 13 TeV, NNLO in QCD with NNLOJET, CT18NNLO, the
G_mu electroweak scheme, mu_R = mu_F = m_ll, seven-point scale variation, over
60 statistically independent runs. Fiducial selection `66 < m_ll < 116` GeV,
`pT_l > 27` GeV, `|eta_l| < 2.5`, applied to Born-level leptons.

Three towers, each an event-level Chebyshev moment as defined in
[`../MOMENT_CONSTRUCTION.md`](../MOMENT_CONSTRUCTION.md):

| tower | map | range | orders |
| --- | --- | --- | --- |
| `mll` | Breit-Wigner | 66 to 116 GeV | 12 |
| `y_abs` | linear | 0 to 2.4 | 12 |
| `pT_ll` | logarithmic, windowed | 30 to 2500 GeV | 20 + window rate |

Columns: `tower, map, lo, hi, order, value, stat, scale`. `stat` is the scatter
over the 60 independent runs, `scale` the half-range over the seven-point
variation. The JSON carries the same numbers plus the window definition and the
window rate with its own two errors.

The recoil window is the C^2 profile `w(pT) = 6t^5 - 15t^4 + 10t^3` with
`t = clip(ln(pT/30)/ln 2, 0, 1)` and a hard cut at 500 GeV; the recoil moments
are `<w T_n> / <w>` and the window rate is `<w> / sigma_fid`.

Produced by `runcards/dy_moments_log30_hi.run`. Regenerate the tables with
`export_dy_moments.py` in the analysis tree.

One caveat worth knowing before you use these: the mass tower is the
statistics-dominated one. Its statistical error exceeds its scale variation for
every order, because the normalised mass shape is nearly scale-independent, so
there is very little scale variation to compare against. The rapidity and recoil
towers are scale-dominated in the bulk.
