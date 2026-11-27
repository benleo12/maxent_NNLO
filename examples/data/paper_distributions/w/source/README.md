# W lepton charge asymmetry and acceptance

The main W plot shows the lepton charge asymmetry at 7 TeV. Both fits use the same production and polar-angle moments. The second also uses recoil–spin moments. The gray dashed prior uses its original LO+PS rate for each charge. The two fitted predictions use the corresponding NNLO charge rates. The middle row divides the fitted asymmetry by independent NNLO in all eleven pseudorapidity bins for each of the three lepton thresholds. The bottom panel divides the baseline prior and fitted predictions by the combined Born-level ATLAS measurement, which uses the 20 GeV cut. The 28 and 32 GeV columns test the predictions after tightening that cut. [ATLAS Table XXVI](https://arxiv.org/pdf/1109.5141)

## Rebuild the two plots

From this directory with the repository at `$REPO`, run

```sh
python3 export_asymmetry.py "$REPO/examples/data/paper_distributions/w/charge_asymmetry.npz"
latexmk -pdf fig_w_asymmetry_v36.tex

python3 export_acceptance.py "$REPO/examples/data/w_predictivity/full_study"
latexmk -pdf fig_w_acceptance_v36.tex
```

The three `w_asymmetry_*.dat` tables and `w_acceptance_32.csv` contain all plotted numbers. Both PDFs have native TeX text and vector artwork. Only fig_w_asymmetry_v36.pdf is intended for the manuscript. fig_w_acceptance_v36.pdf is a numerical check for the supplemental material.

The [original prior table](../lo_ps_prior.csv) gives the full-sample counts, bin cross sections for each charge and the resulting asymmetry. The prior cross sections in the baseline selection are 2829.49 pb for W+ and 1863.89 pb for W−.

The baseline NNLO asymmetry at all seven scales and its numerical errors come from the separate [W validation campaign](../../../w_joint/charge_asymmetry.json). It provides the per-charge bin cross sections and channel-wise resampling errors on their asymmetry. The 28 and 32 GeV NNLO references are in `charge_asymmetry.npz`. The upper panels show the central predictions and the seven-scale envelopes of each fit. The open squares show NNLO with its integration error and the filled circles show the ATLAS data with their experimental errors. In each ratio to NNLO, the fit and the reference vary at the same scale. For ratios to data, only the theory varies. Colored bars show the numerical fit error divided by the central reference. In the NNLO row the black bars give the numerical reference error about one. In the ATLAS panel the gray band gives the experimental error about one.

For the acceptance, the full fiducial sample is the ATLAS 20 GeV baseline, while the second denominator includes the additional W transverse momentum cut at 50 GeV. The numerator applies the same 32 GeV lepton cut within either sample. The LO+PS prior, both fits and the NNLO reference each use their own corresponding denominator. The reference is inclusive NNLO and describes absolute hard-recoil cross sections at NLO. The points show each prediction minus NNLO in percentage points. The prior error combines finite event statistics with the independent NNLO integration error. Each fitted prediction also includes its input-moment uncertainty. The table also supplies the paired change with its covariance since both fits share the input moments and events. The unplotted 27 GeV exploratory failure remains in Appendix C.2.
