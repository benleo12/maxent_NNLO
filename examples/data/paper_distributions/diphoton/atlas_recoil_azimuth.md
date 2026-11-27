# Continuum diphoton production at 8 TeV

[atlas_recoil_azimuth.csv](atlas_recoil_azimuth.csv) gives the ATLAS measurements and both reweighted predictions in the measured bins. The `pt` rows use the photon-pair transverse momentum in GeV. The `dphi` rows use the photon azimuthal separation in radians. Each spectrum is divided by its full fiducial integral. [ATLAS recoil measurement](https://doi.org/10.17182/hepdata.77381.v1/t3) · [ATLAS azimuthal measurement](https://doi.org/10.17182/hepdata.77381.v1/t7)

| Column | Definition |
|---|---|
| `bin_low`, `bin_high` | Lower and upper edges of the measured bin. |
| `data`, `data_err_minus`, `data_err_plus` | Measured density and positive magnitudes of its downward and upward errors in inverse GeV or inverse radians. The published statistical and pointwise systematic errors are combined in quadrature and divided by the full measured integral. Common rate uncertainties cancel. The integral is held fixed in the errors, as the measurement supplies no bin covariance. |
| `in_ratio_range` | One for a measured interval fully inside the plotted range, zero otherwise. All 31 recoil and 40 azimuthal intervals are supplied. |
| `jet_me_*`, `shower_only_*` | Fits to the prior with diphoton+jet matrix elements and the prior with shower radiation, respectively. |
| `*_density`, `*_mc_error` | Reweighted density and its finite-event error at fixed fitted weights. The error includes the shared events in the bin and the full fiducial normalization. |
| `*_ratio`, `*_ratio_mc_error` | Predicted density and its event error divided by the central measured density. The separate data errors above give the experimental band about unity. |
| `*_max_event_fraction`, `*_ratio_shown` | Largest event weight divided by the sum of fitted weights in the bin. The figure draws a ratio when its measured interval is in range and this fraction is below 0.10. |

A density is the sum of fitted bin weights divided by the sum over all selected events and by the bin width. For a bin fraction `f`, its event variance before division by the squared width is the sum of `w² (1 − f)²` inside the bin and `w² f²` outside, divided by the squared full weight sum. Scale and fit-input variations are not included in these errors. The upper spectra in the figure use the original comparison grid, with adjacent shower bins combined where the prior or reweighted sample has fewer well-populated bins.
