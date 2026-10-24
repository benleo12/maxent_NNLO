# eventlevel/: the analysis scripts behind the papers

These scripts made the figures and numbers of the two papers. They are published so that every
number can be traced. They are **not** part of the package API, and they do not run from this
repository alone. They need inputs that are not distributed because of their size:

- the shower samples (`*.npz` in this directory);
- the NNLOJET productions under `$NNLOJET_ROOT` (default `~/nnlojet-v1.0.2`; run cards in
  `../runcards/`);
- the HEPData tables of the ATLAS measurements.

Install the package first (`pip install -e ..`) and run each script from this directory. The
figure → script → settings table is in [`../REPRODUCE.md`](../REPRODUCE.md).

| role | scripts |
|---|---|
| JHEP figures | `explain_matching.py` → `fig_explain_matching_2p.pdf` (`EXPLAIN_LAYOUT=2`; uses `fo_ptll_full.py`); `fig_weight_dists.py` → `fig_weight_dists.pdf`; `fig_transport_law.py` → `fig_transport_law.pdf`; `ggh_eventlevel.py` → `fig_ggh_ptj1.pdf`; `w_eventlevel.py` (two runs) + `fig_w_asym_combined.py` → `fig_w_asym.pdf`; `fig_aa_priors.py` → `fig_aa_priors.pdf`; `fig_aa_eventlevel.py` → `fig_aa_dphi_aa.pdf` (uses `aa_eventlevel_solve.py`, `aa_vs_data.py`); `fig_zj_cross.py` → `fig_zj_cross.pdf` (data from `zj_cross_demo.py`, `zj_cross_err.py`); `fig_snr_spectra.py` → `fig_depth.pdf` (appendix) |
| PRL figures | `fig_dy_spectra.py --imposed-follows` → `fig_dy_imposed_follows.pdf`; `fig_dy_eventlevel.py` → `fig_dy_eventlevel.pdf`; `phistar_prediction.py` → `fig_phistar_prediction.pdf`; `fig_cost_prl.py` → `fig_cost_prl.pdf`; `fig_nnlo_born.py --pulls-strip` → `fig_lepton_pulls.pdf` (optional) |
| numbers quoted in the text | `make_dy_band_weights.py` (the Drell-Yan bands), `fig_map_choice.py` (map comparison), `fig_seam_rule.py` (the matching-scale rule), `aa_angular_test.py` (diphoton angle tower), `zj_eventlevel.py` and `zj_mixed_scan.py` (Z+jet acoplanarity), `fig_covariance_law.py` (the transport-law computation, two-panel form), `moment_bands.py` (first-order band cross-check) |
| studies, not quoted | `convergence_study.py`, `dy_cross_scan.py`, `dy_eventlevel_solve.py`, `decorrelated_prediction.py` |
| event samples | `shower_dy.py`, `make_dy_atlas_npz.py`, `make_dy_atlas_npz_born.py` (Drell-Yan); `shower_w.py` (W); `diphoton_generate_v4.py`, `diphoton_generate_v5.py`, `diphoton_shower_merged.py`, `aa_merge_slices.py`, `aa_v5_pipeline.sh` (diphoton); `build_zj_prior.py` (Z+jet); `ggh_generate_prior.py` (Higgs) |
| shared helpers | `nnlojet_moments.py` (reads the NNLOJET accumulators), `stripper_moments.py` (reads the Stripper NNLO Z+jet moments of the PRL), `bandviz.py`, `pubstyle.py`, `yoda_ref.py` |

All samples are parton level (hadronization off). Multiparton interactions are on for
Drell-Yan, W and Z+jet, and off for diphoton and the Higgs.
