# W lepton charge asymmetry and acceptance

The main W plot compares the two fits with exactly the same boson and polar-angle targets. Only the second fit adds recoil–spin constraints. Each charge is normalized to its fixed-order fiducial cross section. The 20 GeV column uses the combined Born-level ATLAS charge asymmetry at 7 TeV (Aad et al., arXiv:1109.5141, Table XXVI). The 28 and 32 GeV columns use independent NNLO calculations. Neither the asymmetry bins nor the two tighter cut acceptances entered the fits.

## Rebuild the two plots

From this directory with the repository at `$REPO`, run

```sh
python3 export_asymmetry.py "$REPO/examples/data/paper_distributions/w/charge_asymmetry.npz"
latexmk -pdf fig_w_asymmetry_v36.tex

python3 export_acceptance.py "$REPO/examples/data/w_predictivity/full_study"
latexmk -pdf fig_w_acceptance_v36.tex
```

The three w_asymmetry_*.dat tables and w_acceptance_32.csv contain all plotted numbers. Both PDFs have native TeX text and vector artwork. Only fig_w_asymmetry_v36.pdf is intended for the manuscript. fig_w_acceptance_v36.pdf is a numerical check for the supplemental material.

For the asymmetry, both fits are plotted at the central scale. The scale envelopes use all seven predictions. For the two independent comparisons each residual is evaluated with the corresponding scale in the prediction and the independent reference before forming its envelope. The filled ATLAS markers use the published experimental errors. The open NNLO markers use independent integration errors. Colored bars on the lower axes include the response of the fit to the target-integration errors and finite prior statistics. Their covariance with the independent validation is zero. These bars exclude the separately shown reference error. The two fitted curves themselves share target and prior fluctuations.

For the acceptance, the full fiducial sample is the ATLAS 20 GeV baseline, while the second denominator includes the additional W transverse momentum cut at 50 GeV. The numerator applies the same 32 GeV lepton cut within either sample. Both fits and the reference use their own corresponding denominator. The reference is inclusive NNLO and describes absolute hard-recoil cross sections at NLO. All plotted residuals are in percentage points, and bars combine the fit and independent numerical errors. The table also supplies the paired change with its covariance since both fits share the input moments and events. The unplotted 27 GeV exploratory failure remains in Appendix C.2.
