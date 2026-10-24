# NNLOJET run cards

One file per production behind a released table, as run with NNLOJET v1.0.2 and the patches
in `../nnlojet_patch/`. Each production ran one perturbative channel per job: the CHANNELS
block of a shipped card shows one of them, and the cards of the other channels differ only in
that line. Where a card has a `warmup` line, it is the card of the run that made the Vegas grid;
the other runs used the same card without it (production only). `gg_moments2.run` and the two
window-variant cards have none: those productions reused existing grids (see below for the
variants).

| run card | production (run directory) | used for |
|---|---|---|
| `dy_moments_log30_hi.run` | Z at NNLO, μ = m_ll, 298 runs (`dy_profile_log30_hi`) | `moments/dy_nnlo_moments` |
| `w_moments_WP.run`, `w_moments_WM.run` | W± at NNLO, μ = m_ℓν, 40 runs each (`w_moments/WP`, `w_moments/WM`) | `moments/w_nnlo_moments_W{P,M}` |
| `gg_moments2.run` | γγ at NNLO, μ = m_γγ, 40 runs (`gg_moments2`) | `moments/aa_nnlo_moments` |
| `ggh_moments3.run` | on-shell H at NNLO (heavy-top effective theory, `decay_type = 3`), μ = m_H, 40 runs (`ggh_moments3`); also books the leading-jet reference histograms | `moments/ggh_nnlo_moments` |
| `zj_moments.run` | Z+jet at NLO, μ = H_T/2, 40 runs (`zj_moments2`) | `moments/zj_nlo_moments` |
| `ggh_winvar.run` | the H recoil tower with three other matching windows, channels R, RV, RR, 40 runs (`ggh_winvar`) | `moments/ggh_nnlo_moments_winvar` |
| `gg_winvar.run` | the γγ recoil tower with three other matching windows, channels R, RV, RR, 40 runs (`gg_winvar`) | `moments/aa_nnlo_moments_winvar` |
| `dy_ptz_full.run` | the real channels (R, RR, RV) of the Z run with a full-range p_T^ℓℓ histogram, 6 runs (`dy_profile_poc_ptzfull`) | the fixed-order recoil curve below the matching scale in figure 1 of the paper (no table) |

The run directory and card of every table are also recorded in its `provenance` field.

## The window-variant productions

`ggh_winvar.run` and `gg_winvar.run` need the two additional patches
`nnlojet_patch/winvar_*.f90.patch`. They book, in one run:
- the nominal windowed recoil moments (`prof_wpt_0..12`, a check that reproduces the main table);
- the same moments for three other windows, `prof_wpt{A,B,C}_0..12`.

| process | nominal (a, b) | A | B | C | upper edge |
|---|---|---|---|---|---|
| gg → H, p_T^H | (37, 74) | (25, 50) | (50, 100) | (37, 148) | hard cut at 1000 GeV |
| γγ, p_T^γγ | (28, 56) | (19, 38) | (37, 74) | (28, 112) | hard cut at 500 GeV |

Only the channels R, RV and RR run, because LO, V and VV have no recoil and every window gives
them zero weight. The runs use seeds 101–140 and the run names `GGH_WINVAR` and `GG_WINVAR`, so
their output can never be pooled with the main runs by accident. The cards keep only the
accumulators the windowed moments need: 53 single-bin histograms, `norm_born` and the
profiles of the four windows.

The window rate needs the cross section of all six channels. It is therefore W0_V / σ_fid,
with W0_V the pooled `prof_wpt{V}_0` of the variant run and σ_fid the pooled `norm_born` of the
main run (`ggh_moments3`, `gg_moments2`). The variant table's `_runs.json.gz` carries both.

**Vegas grids.** The variant runs did not make their own warm-up. Their grids are copies of the
main runs' grids, renamed to the new run name:

| grid of the variant run | copied from (main run) | md5 |
|---|---|---|
| `ggh_winvar/ch_R/H.GGH_WINVAR.y1.00E-08.R` | `ggh_moments3/ch_R/H.GGH_MOMENTS.y1.00E-08.R` | 88ee94c325e7afa9ababe9d23e27e139 |
| `ggh_winvar/ch_RV/H.GGH_WINVAR.y1.00E-08.RV` | `ggh_moments3/ch_RV/H.GGH_MOMENTS.y1.00E-08.RV` | 895d197d7e4fe3e2023888a99771db20 |
| `ggh_winvar/ch_RR/H.GGH_WINVAR.y1.00E-08.RR` | `ggh_moments3/ch_RR/H.GGH_MOMENTS.y1.00E-08.RR` | d27af58f903c30639aa43bc948a5aabd |
| `gg_winvar/ch_R/GG.GG_WINVAR.y1.00E-09.R` | `gg_moments2/ch_R/GG.GG_MOMENTS.y1.00E-09.R` | 2468ceeebedd4858bb0d15623680d8f0 |
| `gg_winvar/ch_RV/GG.GG_WINVAR.y1.00E-09.RV` | `gg_moments2/ch_RV/GG.GG_MOMENTS.y1.00E-09.RV` | 0e6c5314b40355af2eeb824197ddffce |
| `gg_winvar/ch_RR/GG.GG_WINVAR.y1.00E-09.RR` | `gg_moments2/ch_RR/GG.GG_MOMENTS.y1.00E-09.RR` | f8f2595d08028721019abbd61b9858a1 |

The `ggh_moments3` grids come from its seed-1 warm-up (`warmup = 20000[5]`). The `gg_moments2`
grids are those of an earlier production of the same process and selection (`gg_moments`):
all 40 `gg_moments2` runs were production only.
