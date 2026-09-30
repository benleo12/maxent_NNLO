# Reproducing the figures

Figure -> script -> data for the analyses added on 2026-09-27 (merged diphoton prior, W production).
The earlier figures are produced by the scripts named in `RELEASE_SCRIPTS.txt` of the analysis tree; every script lives in `eventlevel/` here, the NNLOJET run cards in `runcards/`, the moments in `moments/`.
## Added 2026-09-27 (JHEP: merged diphoton prior, W production)

| figure | script | data |
|---|---|---|
| `fig_aa_priors.pdf` | `eventlevel/fig_aa_priors.py` (env `AA_PRIOR_V5`, `AA_WEIGHTS_V5`) | both diphoton prior sets + both weight files |
| `fig_aa_{m_aa,costh_aa,pt_aa,dphi_aa,at_aa}.pdf` | `eventlevel/fig_aa_eventlevel.py` (default prior set `v5`; `AA_PRIOR_SET=v4` for the stress case) | `aa_prior_v5_*.npz` (merged: `diphoton_shower_merged.py` on MG5 `p p > a a` / `p p > a a j` LHE, CKKW-L; box: `diphoton_generate_v5.py --channels gg`; pieces combined with `aa_merge_slices.py`; driver `aa_v5_pipeline.sh`) |
| `fig_w_ptw.pdf`, `fig_w_etal_{WP,WM}.pdf`, `fig_w_asym.pdf`, `fig_w_asym_ctw.pdf`, `fig_w_mt_ctw.pdf` | `eventlevel/w_eventlevel.py` (env `W_CH=LO,R,V,RR,RV,VV`, `W_CHARGES`, `W_CTW_MODE=off|window|joint|full`, `W_FIG_SUFFIX`; the paper figures are the `off` set plus `fig_w_asym` and `fig_w_mt_WP` of the `window` set) | priors `wp_s201.npz`, `wm_s301.npz` from `shower_w.py` on MG5 LO `p p > mu+ vm` / `mu- vm~` (7 TeV, `sm` model, no cuts); NNLOJET `runcards/w_moments_{WP,WM}.run` (one channel per run, six channels, 40 seeds per channel and charge on Perlmutter); the resulting moments are released as `moments/w_nnlo_moments_{WP,WM}.{json,csv}` (`export_w_moments.py`) |

The W towers require the regenerated `nnlojet_patch/` (W blocks in `Observables.f90`, the decay-angle
functions in `EvalFuncs.f90`); `make nnlojet_w` builds the process library.

## Released moment tables (2026-09-29)

`moments/*.json` and `.csv` for Drell-Yan (298 runs), W+ and W- (40 runs per charge), diphoton (40),
gluon-fusion Higgs (40) and Z+jet at NLO (40) are produced by `export_moments.py` from the NNLOJET
productions listed in `runcards/`; `moments/load_moments.py` turns a table into the `moments` and
`config` inputs of `upgrade`, and `examples/upgrade_with_released_moments.py` applies them to a
user's prior. Round-trip check: the Drell-Yan and W tables reproduce the paper's solves (effN,
closure, moment counts; weights to 1e-5 for W).
