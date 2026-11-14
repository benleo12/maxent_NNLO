# Compact W event inputs for the constraint-study workflow

`w_wp_prior.npz` and `w_wm_prior.npz` each contain 80,000 actual shower events.
They were selected before the new study fits, by a uniform fixed hash rule from
the existing baseline-selected W+ and W− samples (522,365 and 490,046 events).
They provide compact demonstrations with different event statistics from the
full paper study; they are subsets of those productions.

The [generation inputs](../../../runcards/w_prior/README.md) include process/run/
parameter cards and the original shower extractor. The [exporter](../../export_w_fixture.py)
defines the exact selection: order the SHA256 values of
`w-public-v1:731131:charge:original_row_id`, retain the lowest 80,000, then restore
original event order. There are no kinematic quotas or fit-dependent choices.

Each JSON companion records the parent-file hash, original/generated/selected
counts, generator flags, normalization, inclusion probability and fixture hash.
Both original pre-QED four-vectors are retained as `(px, py, pz, E)` in GeV.
`event_id` and `source_index` are original row IDs, unique within each charge.
`original_weight`, `inclusion_probability`, `inverse_inclusion_probability` and
`weight = original_weight / inclusion_probability` keep the sampling correction
explicit. The inverse inclusion factor is constant within a charge, so it
cancels when normalizing a prior shape.

The coordinate columns are `m_w`, `y_w`, `y_abs`, `pt_w`, `c`, `ctw1=c+1`,
`sx`, `sy`, `rho`, `rho_sx`, `rho_sy`, `pt_l`, `pt_nu`, `mt`, `y_l` and `eta_l`.
The [coordinate builder](../../w_coordinates.py) documents the signed massless
Cartesian Collins–Soper convention. `sx`/`sy` have no physical azimuthal meaning
at zero recoil; the corresponding `rho_sx`/`rho_sy` functions have a continuous
zero limit. The charged lepton defines the angle for both charges.

The baseline is pT_l>20 GeV, pT_nu>25 GeV, mT>40 GeV, |y_l|<2.5 and the requested
charge. `eta_l` is signed pseudorapidity computed from the retained four-vector;
`y_l` is rapidity. Their tiny muon-mass difference is retained. No prior stored
25 GeV fiducial flag is used to make this selection.

The metadata includes counts and effective counts in coarse recoil, absolute
angle and absolute rapidity cells. These reveal sparse regions of the compact
sample; they do not certify phase-space support. The workflow's independent-row
errors describe the unconditional generator sample, not conditional resampling
uncertainty of a fixed parent pool. The uniform subset is intentionally used
instead of fixed kinematic quotas to keep that interpretation simple.

Absolute posterior event weights come from the fitted target cross section.
The prior `weight` array itself is not already an absolute pb-per-event
posterior: its original generation normalization is recorded separately.
