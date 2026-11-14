# Physical interpretation to use after independent comparison

The new study separates a change in the recoil prescription from additional
production–decay information. The original anchor retains the previous blended
recoil targets. R imposes direct windowed recoil moments. H adds five
reflection-even spin-one functions in the hard recoil window, each multiplied
by the first three rapidity polynomials. S adds continuous recoil-suppressed
versions over the full fiducial phase space. R minus the blended anchor is a
prescription comparison; H minus R and (R+H+S) minus H isolate the effects of
the added blocks under their stated prescription. S is global, including recoil
above500 GeV; its effect is not exclusively a low-qT correction.

Training selects H and S for both charges, with the same choices in both fixed
seed halves. D and M do not meet the all-scale addition rule, both before and
after H/S. This does not show that higher polar moments or mass–angle
correlations are absent. Those are specific finite blocks with a specified
statistical threshold. The narrowed/gated/widened window controls and the
unconditioned H0/S0 controls remain comparisons, not retrospectively selected
solutions.

All H/S functions vanish at Born level. An inclusive NNLO input therefore
supplies NLO information for them. They are cut-weighted moments, not universal
angular-coefficient projections: the fiducial acceptance distorts the usual
orthogonality relations. Neither the hard-window targets nor the continuous
suppressed moments establish resummed recoil accuracy.

For each old exploratory pT cut, let I be the physical lepton-cut indicator and
I0 the indicator of b=(m/2)sqrt(1-c²)>cut. On the same originally selected events,
I=I0+(I-I0) exactly. The posterior-minus-fixed-order acceptance difference is
therefore the sum of a projected-population difference and a recoil-migration
difference. Each sample uses its own original baseline normalization. The
common covariance must accompany both terms because they can cancel strongly.
The migration term combines transverse boost and decay-azimuth correlations;
it is not an isolated shower error. The projected term also inherits the
original fiducial selection and is not an alternate Born-level sample.

The3×3 qT–|c| ledger uses the prespecified regions qT<25,25≤qT<50,qT≥50 and
|c|≤.6,.6<|c|<.7,|c|≥.7. A map of accepted baseline fractions compares actual
independent fixed-order counts in these booked regions. It differs from the
old prior-only redistribution map, which had no2D fixed-order reference.

The prespecified new tests are pTl>28, pTl>32, mT>60, pTl>32 with qT>50,
and mT>60 plus pTnu>30 plus|eta_l|<1.7. For the hard test report both its
baseline-normalized rate and epsilon(pTl>32 | qT>50). The28 GeV test probes
kinematics close to the chosen mass gate's flat endpoint;32 GeV, hard recoil
and the combined cuts provide complementary checks. Historical spectra made
some of these shapes available before this study; the new independent
integrations and frozen protocol establish the current comparison chronology.

The exact booked lepton coordinate is y_l. It is conventionally labeled eta_l
under the massless-lepton convention, where y_l=eta_l. The retained massive
shower four-vectors are evaluated consistently using y_l. The audited muon-mass
and coordinate-reconstruction effects are far below the percent-level effects
under investigation.

## A directly interpretable measured spin moment

In the massless convention, rho*s_x=(pTl²-pTnu²)/(m²+qT²). Before adding S,
the R+H posterior minus the fixed-order training moment is
−0.002986716±0.000144331 for W+ and+0.001910463±0.000151090 for W−.
These errors use the complete training residual covariance, including fitted
response and finite-prior statistics. All seven coherent-scale differences
have the same sign within each charge: W+ spans[−0.003350683,−0.002595566]
and W−[+0.001759032,+0.002055332]. The S constraint therefore increases the
normalized charged-lepton versus neutrino pT² imbalance for W+ and decreases
it for W−. This is a resolved physical deficiency of that training anchor;
it does not itself establish threshold-acceptance closure or identify a
uniquely low-qT origin. Source labels, ordering and hashes are saved in
IMBALANCE_TRAINING_DIAGNOSTIC.json.

More generally, defining E_T²=m²+qT², rho=qT/E_T, c=cos(theta),
s_x=sin(theta)cos(phi), and d=sin²(theta)cos(2phi), exact massless kinematics give
4pTl²/E_T²=1-c²+(rho²/2)(1+c²+d)+2rho*s_x. The recoil terms contain the same
rho*s_x, rho²*d and rho²*P2(c) structures measured by S. E_T here is the boson
transverse energy, distinct from the leptonic transverse-mass observable.
