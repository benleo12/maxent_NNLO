# Full-sample differential distributions

Each link opens a figure from the paper or its binned predictions. The runnable
examples fit compact event samples. These figures use the full production samples.

| Application | Figure | Binned predictions |
|---|---|---|
| Neutral-current Drell–Yan | [Dilepton mass, absolute rapidity and angular recoil](drell_yan/figure.pdf) | [CSV](drell_yan/distributions.csv) |
| Charged-current Drell–Yan | [Lepton charge asymmetry at three thresholds](w/figure.pdf) | [CSV](w/predictions.csv), [baseline data and residuals](w/baseline.csv), [scale members and covariances](w/charge_asymmetry.npz) |
| Gluon-fusion Higgs | [Recoil and jets](higgs/jets.pdf), [diphoton recoil and rapidity after photon cuts](higgs/photons.pdf) | [Fiducial spectra, variations, inclusive recoil and jet counts](higgs/fiducial.npz) |
| Continuum diphotons | [Photon-pair mass and pseudorapidity separation](diphoton/figure.pdf) | [Densities, reference and finite-event uncertainties](diphoton/distributions.npz) |

The resolved-jet application has a [separate benchmark and refinement study](../zj_refinement/RESULTS.txt).

## Using a saved fit

From the repository root, run

```bash
python examples/real_drell_yan.py --output dy_demo --plot
python examples/plot_result.py dy_demo/posterior.npz phistar \
    --range 0.001 3 --log-x --log-y --output dy_demo/phistar.pdf
```

The first command fits the bundled events. The second uses their saved weights to
plot an observable built from the two lepton directions, with its scale band.
Replace `phistar` with another numeric observable stored per event to use the same fit.
The [example catalogue](../../README.md) gives the corresponding commands for the other processes.

## Reading the numbers

- **Drell–Yan.** Each CSV row gives a density and its bin edges. The mass
  distribution uses the fiducial cross section, and each rapidity curve has unit
  area over the displayed range through 2.4. The angular spectrum uses the full
  fiducial normalization, including events outside the displayed recoil range.
  Scale endpoints refer to the posteriors. The measured angular spectrum retains
  the ATLAS bins and total pointwise errors. [ATLAS Table 4b](https://doi.org/10.17182/hepdata.92377.v3/t5)
- **W.** The cuts in the three columns are 20, 28 and 32 GeV. The first comparison
  uses the combined ATLAS Born-lepton measurement. The other two use an independent NNLO
  integration. The NPZ includes the seven scale members, the separate covariances
  for integration and prior statistics, and differences between fits evaluated on
  the same events. The `models`, `pairs` and `scale_names` arrays give their
  order. [ATLAS Table XXVI](https://arxiv.org/pdf/1109.5141)
- **Higgs.** The fiducial spectra use absolute units, fb/GeV for transverse
  momenta and fb per unit rapidity. Only gluon fusion is reweighted. Each total
  includes the same VBF, WH, ZH and ttH contributions. The first leading-jet bin
  represents events without a selected jet above 30 GeV. The `metadata_json`
  field specifies the rate, scale, transition and finite-event uncertainties.
  The simulation applies isolation using the transverse energy of the summed track
  four-vector. The ATLAS paper specifies a scalar sum of track transverse momenta.
  The stored jets use visible final-state particles. The ATLAS particle-level
  definition treats prompt leptons and invisible decay products differently and
  removes jets close to selected electrons. These differences also enter the
  comparisons with matched generators.
  [ATLAS measurement](https://arxiv.org/abs/2202.00487) · [HEPData record](https://doi.org/10.17182/hepdata.137886.v1)
- **Continuum diphotons.** The file contains normalized densities for merged
  (`v6`) and shower-only (`v4`) events, before (`p`) and after (`q`) reweighting.
  The `m` arrays use photon-pair mass in GeV. The `eta` arrays use the
  hyperbolic tangent of half the absolute photon pseudorapidity difference.
  ATLAS used fixed-cone isolation. The simulations and fixed-order mass prediction
  use smooth-cone isolation. [ATLAS mass spectrum](https://doi.org/10.17182/hepdata.77381.v1/t2) · [ATLAS angular spectrum](https://doi.org/10.17182/hepdata.77381.v1/t6)

All NPZ archives load with `numpy.load(path, allow_pickle=False)`.
File digests are listed in [SHA256.json](SHA256.json).
