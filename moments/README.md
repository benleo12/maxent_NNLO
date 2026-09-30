# Fixed-order moments, released for use with your own prior

These tables are the fixed-order input behind every application of the paper.
Each JSON file is self-describing; `load_moments.py` turns it into the two
inputs of `maxent_upgrade.upgrade()`:

```python
import sys; sys.path.insert(0, "/path/to/maxent_NNLO")
from maxent_upgrade import upgrade
from moments.load_moments import load

moments, config = load("dy")              # the constraint set used in the paper
result = upgrade(events, moments, config)  # events: your prior, see below
```

`python moments/load_moments.py` prints, for every table, the arrays your
`events` dict must carry and the constraint set the paper imposed. A complete
bring-your-own-prior script is `examples/upgrade_with_released_moments.py`.

| table | process | order | runs | towers imposed in the paper | also in the table |
| --- | --- | --- | --- | --- | --- |
| `dy_nnlo_moments` | pp -> Z/gamma* -> l+l-, 13 TeV, ATLAS fiducial | NNLO | 298 | `mll` (BW, 12), `y_abs` (lin, 12), `pT_ll` window above 30 GeV (20 + rate) | `costh` (6), joint `mll_x_costh` (16), lepton-pT reference towers |
| `w_nnlo_moments_WP`, `_WM` | pp -> W+- -> l nu, 7 TeV, ATLAS fiducial | NNLO | 40 per charge | `m_w` (BW, 12), `y_abs` (lin, 12), `pt_w` window above 25 GeV (20 + rate) | decay-angle window `ctw_window` (6 + rate), full-range `ctw_full` (8), joint `pt_w_x_ctw` (16) |
| `aa_nnlo_moments` | pp -> gamma gamma, 8 TeV, ATLAS fiducial | NNLO | 40 | `m_aa` (log, 6), `costh_aa` (lin, 6), `pt_aa` window above 28 GeV (12 + rate) | |
| `ggh_nnlo_moments` | pp -> H (gluon fusion), 13 TeV, inclusive | NNLO | 40 | `y_abs` (lin, 6), `pT_H` window above 37 GeV (12 + rate) | |
| `zj_nlo_moments` | pp -> Z(-> l+l-) + jet, 13 TeV, ATLAS fiducial | NLO | 40 | `mll` (BW, 6), `y_abs` (lin, 6), `ptj1` and `ptj2` windows above 30 GeV, joint `ptj1_x_ptj2` (16) | `pimdphi` window (12), used to 4 orders in the paper |

## What your prior must provide

`events` is a dict of equal-length arrays: the observables named by the table
(the `observable` field of each tower, printed by `load_moments.py`) and the
strictly positive prior weights under `weight`. Both sides must integrate the
same phase space: restrict your events to the table's `fiducial` selection,
with the observables defined as there, or use the inclusive Higgs table with
an inclusive prior. Two definitions matter most:

* **Born-level leptons.** Every lepton observable is defined with the QED
  final-state radiation added back to the lepton (the "Born" or "dressed with
  all its own photons" lepton), as in a QCD-only fixed-order calculation. Bare
  leptons make a mass tower spend its resolution undoing radiation
  (section 4 of the paper).
* **Decay angles** are Collins-Soper angles in the boson rest frame:
  `costh` is the signed angle of the negative lepton for the Z, `costh_aa` the
  absolute value for the photon pair, and for the W the table uses
  `ctw1 = cos(theta*) + 1` of the charged lepton so that the window profile
  can be written on a positive variable.

A prior that lacks the channels the corrections populate pays in effective
statistics: the paper's diphoton section shows a Pythia 2 -> 2 prior retaining
4% of its effective count against 94% for a merged gamma gamma + jet prior.

## Schema

One schema for every table. `towers[key]` carries `kind` (`born`, `window`, or
`reference` for towers released for validation but never imposed), the
`observable` name, `map` (`lin`, `log`, `bw` with `bw = [M, Gamma]`), `range`,
`orders`, `values`, `stat_errors` (run-to-run scatter divided by the square
root of the number of runs) and `scale_errors` (half-range over the
seven-point variation). Window towers add `soft_lo`, the `profile`
`{a, b, c[, d]}` of the C^2 window `w = 6t^5 - 15t^4 + 10t^3`,
`t = clip((ln x - ln a)/(ln b - ln a), 0, 1)`, and the window rate
`<w>/sigma_fid` with its two errors. `joint[key]` carries the mixed moments
`<w_x w_y T_m(u(x)) T_n(u(y))>/<w_x w_y>` keyed `"m,n"` with their errors and
rate. `default_constraints` lists what the paper imposed. `sigma_fid_pb` is the
fixed-order fiducial cross section with its statistical and scale errors; the
loader normalizes nothing, and `examples/upgrade_with_released_moments.py`
uses it to set the absolute rate of the upgraded sample.

Moment definitions: Born `mu_n = <T_n(u(v))>` over the fiducial region,
window `mu_n = <w T_n(u(x))>/<w>`, both pooled over the independent runs as
the ratio of summed numerators to summed denominators. The maps are
`u_lin = 2(v - lo)/(hi - lo) - 1`, `u_log = 2(ln v - ln lo)/(ln hi - ln lo) - 1`
and `u_BW = 2(t(v) - t(lo))/(t(hi) - t(lo)) - 1` with
`t(v) = arctan((v^2 - M^2)/(M Gamma))`.

## Caveats

* **Higgs normalization.** The `norm_born` accumulator of the Higgs production
  carries an overall normalization that does not reproduce the known inclusive
  cross section (about 600 times too large). The shape moments are ratios and
  are unaffected, and the paper uses only those; `sigma_fid_pb` is `null` for
  this table, and an upgraded sample should be normalized to a cross section
  taken from elsewhere.
* **The Higgs table is fully inclusive.** Use it with an inclusive prior and cut
  afterwards; the acceptance of any later cut then carries shower accuracy.
* **Reference towers** (lepton pT for Drell-Yan, the full-range W decay angle)
  are released so that a user can check predictions against fixed order. They
  are not constraints, and the paper shows why the full-range decay angle
  should not be imposed.
* **The map is part of the target.** The loader reproduces the maps compiled
  into the fixed-order runs; do not change them. A different map converges
  with clean closure onto wrong constraints.

All tables were produced by `export_moments.py` in the analysis tree from the
NNLOJET runcards in `../runcards/`. The Drell-Yan, W+ and W- tables reproduce
the paper's solves through the loader event by event, every weight agreeing to
better than 1e-4.
