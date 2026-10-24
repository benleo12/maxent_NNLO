# Reproducing the papers

## Prerequisites

Each step needs:
- the package installed: `pip install -e .`;
- the NNLOJET productions under `$NNLOJET_ROOT`. They were made with `nnlojet_patch/` and the
  run cards in `runcards/`;
- the shower samples, produced by the scripts in `eventlevel/` and not distributed.

Run every script from `eventlevel/` with the samples in that directory, and set
`OPENBLAS_NUM_THREADS=1`.

The released moment tables come from the same productions:
- `python export_moments.py` regenerates them;
- `python tests/test_released.py` checks them.

## JHEP paper (maximum-entropy moment matching for color-singlet processes)

| figure | script | settings | inputs |
|---|---|---|---|
| `fig_explain_matching_2p.pdf` | `explain_matching.py` | `EXPLAIN_LAYOUT=2` | `dy_prior_atlas_v3.npz`; `dy_profile_log30_hi`; `dy_profile_poc_ptzfull` (run card `dy_ptz_full.run`, through `fo_ptll_full.py`) |
| `fig_weight_dists.pdf` | `fig_weight_dists.py` | | `dy_prior_atlas_v3.npz`; `dy_band_weights.npz` (from `make_dy_band_weights.py`); `dy_minnlo_atlas_v4.npz`, `dy_mcatnlo_atlas_v4.npz`, `dy_powheg_atlas_v6.npz` |
| `fig_transport_law.pdf` | `fig_transport_law.py` | | `dy_prior_atlas_v3.npz`; `dy_profile_log30_hi` |
| `fig_ggh_ptj1.pdf` | `ggh_eventlevel.py` | | `ggh_prior_v2_s1.npz`, `ggh_prior_v2_s2.npz`; `ggh_moments2`; band cache `ggh_band_weights.npz` |
| `fig_w_asym.pdf` | `w_eventlevel.py` twice, then `fig_w_asym_combined.py` | `W_CH=LO,R,V,RR,RV,VV W_CHARGES=WP,WM`, with first `W_CTW_MODE=off W_FIG_SUFFIX=_nnlo`, then `W_CTW_MODE=window W_FIG_SUFFIX=_nnlo_window` | `wp_s201.npz`, `wm_s301.npz` (`W_PRIOR_DIR`); `w_moments/{WP,WM}`; ATLAS 7 TeV W asymmetry |
| `fig_aa_priors.pdf` | `fig_aa_priors.py` (or `bash aa_v5_pipeline.sh priors`) | `AA_PRIOR=$V5 AA_PRIOR_V5=$V5 AA_WEIGHTS_V5=aa_eventlevel_weights_v5.npz` | `aa_prior_v5_*.npz` (merged prior) and `aa_prior_v4_*.npz` (2 → 2 prior); weights from `aa_eventlevel_solve.py` |
| `fig_aa_dphi_aa.pdf` | `fig_aa_eventlevel.py` | `AA_PRIOR=$V5 AA_WEIGHTS=aa_eventlevel_weights_v5.npz AA_FIG_SUFFIX=""` | the same priors; `gg_moments2`; `atlas_aa_8tev.npz`; band cache |
| `fig_zj_cross.pdf` | `fig_zj_cross.py` | | the sixteen rows are embedded; `zj_cross_demo.py` and `zj_cross_err.py` regenerate them from `zj_prior.npz` (≥ 1 jet) and `zj_moments2` |
| `fig_depth.pdf` (appendix) | `fig_snr_spectra.py` | | `dy_prior_atlas_v3.npz`; `dy_profile_log30_hi` (298 runs) |
| `fig_code_flow.pdf` (appendix) | `docs/flowchart.tex` | | |

Numbers quoted in the text without a figure:
- the Drell-Yan scale and bootstrap bands: `make_dy_band_weights.py`;
- the map comparison: `fig_map_choice.py`;
- the matching scales: `fig_seam_rule.py`, or `maxent_upgrade.matching_scale`;
- the diphoton angle tower: `aa_angular_test.py`, with `AA_PRIOR_SET=v4` for the 2 → 2 prior;
- the Z+jet acoplanarity: `zj_eventlevel.py`.

The effective statistics, closures and resolution spectra of every application also come out
of `maxent_upgrade.load_moments(process)` and `upgrade()` on the same priors (the round-trip
checks in `moments/README.md`).

## PRL (Assi, Poncelet, Whitehead)

| figure file | script and flag | content |
|---|---|---|
| `fig_dy_imposed_follows.pdf` (Fig. 1) | `fig_dy_spectra.py --imposed-follows` | what is imposed closes (\|y_ll\|), what is never imposed follows (p_T^ℓ1) |
| `fig_dy_eventlevel.pdf` (Fig. 2, left) | `fig_dy_eventlevel.py` | p_T^ℓℓ against ATLAS, with the NLO and NNLO (Z+jet) recoil inputs |
| `fig_phistar_prediction.pdf` (Fig. 2, right) | `phistar_prediction.py` | φ*_η against ATLAS |
| `fig_cost_prl.pdf` (End Matter) | `fig_cost_prl.py` | cost to reach an effective sample size |
| `fig_lepton_pulls.pdf` (End Matter, optional) | `fig_nnlo_born.py --pulls-strip` | pulls of the never-constrained lepton-p_T moments |

Without the flags, `fig_dy_spectra.py` writes the single-observable spectra `fig_dy_mll`,
`fig_dy_yll`, `fig_dy_ptl1` and `fig_nnlo_born.py` writes `fig_nnlo_pt_l{1,2}_moments` and `fig_nnlo_yll`, the
figures of earlier versions.  The bands are drawn by `bandviz.py` (`step_band`, and
`draw_series_unc(..., edge_to_edge=True)` for whole-bin bands; the default keeps the
centre-to-centre fill of the JHEP figures).

The PRL's NNLO Z+jet recoil variant reads the Stripper moments through `stripper_moments.py`
(`DY_RECOIL_XML`); those moments are not part of this repository.

## Event samples

| process | how the sample was made |
|---|---|
| Drell-Yan | `shower_dy.py` on MG5_aMC LO events, then `make_dy_atlas_npz_born.py` (ATLAS fiducial, Born-level leptons) |
| W | `shower_w.py` on MG5_aMC LO `p p > mu+ vm` / `mu- vm~` at 7 TeV |
| Z+jet | `build_zj_prior.py` (anti-k_T R = 0.4 jets from the Drell-Yan shower) |
| diphoton, merged prior | `diphoton_shower_merged.py` on MG5 `p p > a a` / `p p > a a j` (CKKW-L), plus the Pythia gg box from `diphoton_generate_v5.py --channels gg`, combined with `aa_merge_slices.py`; driver `aa_v5_pipeline.sh` |
| diphoton, 2 → 2 prior | `diphoton_generate_v4.py` |
| gluon-fusion Higgs | `ggh_generate_prior.py` (Pythia 8 `HiggsSM:gg2H`, H stable, 13 TeV, inclusive) |

All samples are Pythia 8.312 at parton level: hadronization is off. Multiparton interactions
are on for Drell-Yan, W and Z+jet, and off for diphoton and the Higgs.
