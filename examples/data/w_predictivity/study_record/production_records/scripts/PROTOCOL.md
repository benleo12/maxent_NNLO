# W predictivity follow-up

Started 2026-11-14. The published v17 study in sibling `w_validation` is immutable.
This follow-up investigates the residual lepton-acceptance differences; it does
not assume that additional moments must remove them.

## Frozen new validation definitions

These definitions were written before producing or opening any new reference.
Old lepton and mT spectra were already public: these are prespecified tests on
new independent integrations, not a claim that their underlying shapes were
historically unavailable.
The baseline remains 7 TeV, charged-lepton pT > 20 GeV, neutrino pT > 25 GeV,
mT > 40 GeV and |eta_l| < 2.5, separately for W+ and W-.

1. Fractions of that baseline passing charged-lepton pT > 28 and > 32 GeV.
2. For pT_l > 32 GeV and qT > 50 GeV, both the fraction of the baseline and
   the conditional fraction within the qT > 50 GeV baseline subset.
3. The fraction passing mT > 60 GeV, with every other baseline cut retained.
4. Charge asymmetry in the same eleven absolute-lepton-eta bins as v17,
   for the 28 GeV, 32 GeV and mT > 60 GeV selections.
5. A combined fiducial fraction with mT > 60 GeV, neutrino pT > 30 GeV and
   |eta_l| < 1.7, retaining the baseline charged-lepton threshold of 20 GeV.

The old 27/30 GeV cuts and old lepton spectra are exploratory evidence only.
They may motivate diagnostics but are never described as independent validation
for this follow-up. New predictions will use independent integration seeds and
will remain unopened until the training choice is frozen. No direct indicator
of a new validation cut is an allowed fitted constraint.

## Initial diagnostics, before new production

Reconstruct the CS tetrad and exact charged-lepton pT from existing lepton and
neutrino four-vectors. Fix the x-axis along the boson transverse momentum and
fold z along its longitudinal direction, reflecting y to retain handedness.
At qT=0, the azimuthal axis is arbitrary; observables involving it must vanish
continuously or have a hard-recoil window.

Audit sensitivities to invariant mass, recoil, polar angle and azimuth on the
same baseline events. Counterfactual kinematic changes retain that original
denominator; they are sensitivity diagnostics, not physical alternate samples.
Inspect old 27/30-cut migrations in qT ranges [0,25),[25,50),[50,infinity) and
in the flat, transition and outside regions of the old angular window.

New exploratory reference bookings will include old-cut intersection sums in
these fixed regions, if practical. Their absolute contributions can localize
an actual fixed-order difference, unlike prior-only migration sensitivities.
Low-qT fixed-order differences are not automatically errors of the shower.

## Controlled candidate design, to be finalized before production

Keep the v17 selected fit as the anchor. Compare separate additions/controls:
unchanged-window angular degree, a physically justified window change at fixed
degree, mass-polar products, and infrared-safe recoil/azimuth functions motivated
by spin-one decay. Refit the anchor constraints for each candidate and retain
paired prediction differences. Effects of overlapping blocks need not add.

Fiducial cuts distort harmonic projections. Harmonic averages are bounded
measurement functions here, not the standard uncut angular-coefficient estimates.
Wider polar windows must respect the mass-dependent Born edge. Hard-window
harmonics isolate information supported by W+jet fixed order. Recoil-suppressed
harmonics may test broader moments but do not supply a new resummed prediction.

The exact candidate blocks, selection statistic, training precision and seed
allocation will be frozen in a production addendum after the initial kinematic
audit and independent design review. They will not use new validation outputs.

## Reproducibility and resources

All new files and remote jobs use w_predictivity directories. Reuse integration
grids; preserve every physical seed and record exclusive task ownership. Use
validated NNLOJET seeds below one million, disjoint from earlier campaigns.
Heavy work runs on Perlmutter/Symmetry or the mini, never the laptop. No paper,
public-repository or Overleaf writes are performed by this study agent.

## Primary physics references

- Spin-one angular structure and fiducial treatment:
  https://link.springer.com/article/10.1140/epjc/s10052-022-10641-1
- Fiducial recoil power corrections and exact lepton kinematics:
  https://arxiv.org/abs/2006.11382
- W angular-coefficient measurement:
  https://arxiv.org/abs/2509.13759

These motivate the diagnostics; none establishes the source of this sample's
remaining discrepancy in advance.

## Exact production addendum (design-review stage)

All targets below are expectations over the original baseline fiducial region.
Each run books a common vector and all seven scales. The vector includes both
fit functions and diagnostic functions, preserving their complete covariance.
Exploratory old-cut indicators are booked separately and excluded from every
fit and every model-selection statistic.

The anchor is refitted with the new common training integration, keeping its
v17 production orders and charge-dependent angular/joint degrees. Thus anchor
and additions share the same target fluctuations. Original v17 weights are
retained as a separate historical reference, not silently replaced.

A second anchor R uses windowed recoil functions w(qT)T_n and the window
probability w, with direct fixed-order targets from the same vector. It replaces
the blended recoil tower and its above-cut prior fraction, retaining all other
orders. This is the package's existing prescription for explicit joint recoil
constraints. In the transition, a direct <w h>FO target is not generally the
same functional of c(1-w)p+wFO, so H/S must not silently combine those distinct
prescriptions. H/S and the combined sequence use R. R minus the blended anchor
is reported separately. D/M/window controls are run on both anchors. All scales
must pass feasibility, convergence and target-residual checks without silently
dropping a target or altering its definition.

Let c be exactly the v17 massless signed-CS definition, with z folded along
the boson longitudinal direction. Define x along qT, and reflect y along with
z. The Cartesian decay components are sx=sin(theta)cos(phi) and
sy=sin(theta)sin(phi), computed by four-vector contraction without an angle
inversion. The small muon-mass deviation between these massless components
and a unit rest-frame direction is documented by the kinematic audit.
At qT=0 use the continuous limiting values for recoil-suppressed functions.

Define uy=clip(|yW|/2-1,-1,1), um as the existing Breit-Wigner map on [40,1000]
with M=80.385, Gamma=2.085, rho=qT/sqrt(m^2+qT^2), and
P2=(1-3c^2)/2. Let wH be the existing recoil window (25,50,500 GeV).

The added training functions are:

- Degree block D: old window wc times T_n(c), n=7,...,10 (4 functions).
- Mass block M: wc T_i(um)T_j(c), i=1,2 and j=0,1,2 (6 functions).
- Hard spin block H: wH times {P2,c,2c sx,sx^2-sy^2,sx}, each multiplied
  by T_k(uy), k=0,1,2 (15 functions).
- Suppressed spin block S: {rho^2 P2,rho^2 c,2rho c sx,
  rho^2(sx^2-sy^2),rho sx}, each times T_k(uy), k=0,1,2 (15 functions).

H and S vanish at Born level. Inclusive NNLO supplies them through the same
order as NLO W+jet, not NNLO W+jet. S is continuous and infrared safe but
does not provide resummation. Sine harmonics are omitted because the chosen
cuts/observables are even under reflection of the decay azimuth; no statement
about their general vanishing angular coefficients is made.

For the window controls define b=(m/2)sqrt(max(1-c^2,0)) and a common gate
g=S5(clip(log(b/25)/log(28/25),0,1)), with S5(t)=6t^5-15t^4+10t^3.
Set g=0 for b<=25, including the entire region m<50 GeV. This gate excludes
the transverse-cut Born edge; it does not remove rapidity acceptance effects
or establish validity of every differential fixed-order prediction.
The 28 GeV held-out lepton cut intentionally probes kinematics near this
gate's plateau boundary. It remains absent from the fitted functions, but is
not presented as equally independent of feature design as the complementary
32 GeV, conditional hard-recoil and combined fiducial tests.

Three windows use the same g, with the existing logarithmic-in-(1+c) profile:
N: flat |c|<=.5, zero |c|>=.6;
G: flat |c|<=.6, zero |c|>=.7 (gated-old control);
W: flat |c|<=.65, zero |c|>=.75.
For each window book w T_n(c), n=0,...,6, and
w T_k(uy)T_n(c), k=1,2 and n=0,1,2 (13 functions/window).
The n=0 window probability is always imposed once. Each window-control fit
replaces the anchor window consistently in its angular and existing joint
functions. G versus anchor diagnoses the mass gate; N/W versus G diagnoses
width at the same gate and degree. It is not described as a pure width change
when comparing G directly with the ungated anchor.

All candidate fits are fixed before reference predictions are opened:
anchor, D, M, H(k=0), H(k<=2), S(k=0), S(k<=2), N, G, W, and the existing
degree-four rapidity-angle completion as a correlation control. Singleton
additions always retain and refit the anchor constraints.

One optional combined candidate starts from R and uses the following fixed
order, independently for each charge: D, M, H(k<=2), S(k<=2). At each step compare that complete
block's residual to the training integrals, including fitted-response and
finite-prior covariance. Add it only when the approximate chi-square p-value
is below .01 at all seven coherent scales. Use standardized residual covariance
and retain eigenvalues above 1e-6 of its largest eigenvalue. Do not select
window controls or reference cuts using this rule. Record all decisions,
ranks, conditioning, target residuals and all candidates, even failures.
The rule is a finite-integration diagnostic, not calibrated theory coverage.

Training starts with 64 independent 250k-point runs per channel and charge,
subject to pilot runtime. Check covariance rank/conditioning and compare the
two fixed seed halves for block stability before freezing choices. If selected
blocks differ between halves or numerical target errors preclude diagnosis,
extend training only, preserving every seed and the rule. Such an extension
must be documented before independent validation is opened.
Only two doublings are allowed, to 128 and then 256 runs per channel. At that
cap, remaining split-half instability makes the combined selection inconclusive;
all predefined singleton results are retained. No extension is made to seek a
particular selected block or favorable independent prediction.

Task-size amendment before any scientific production: measured training runtime
favors 100k-point tasks. Preserve the exact information budgets above by using
160 independent seeds initially, with at most two doublings to 320 and 640.
Each level still has 16M, 32M or 64M integration points per channel and charge.
Reference tasks retain 250k points, initially 64 seeds per channel/charge.
No precision budget was reduced to fit a scheduler limit. New physical seeds
are explicitly disjoint from all 13,888 runs of v17 and below NNLOJET's limit.

Exploratory old-cut localization books baseline, pTl>27 and pTl>30 rates in
the nine intersections of qT=[0,25),[25,50),[50,infinity) with |c|=[0,.6],
(.6,.7),[.7,1], and separately qT with m=[0,76),[76,84),[84,infinity) GeV.
These disjoint sums add to the total accepted rate and can localize an actual
difference. They never select fitted features. Their low-qT fixed-order
components remain diagnostic comparisons, not an instruction to replace
resummed shower behavior.

For each old cut (27,30) and each of the nine recoil/polar-window regions,
also book I0=1[b>cut] on exactly the same original baseline events. Together
with physical I=1[pTl>cut], this gives the exact identity I=I0+(I-I0).
The posterior-minus-FO accepted fraction can therefore be separated into a
Born-projected population contribution and a recoil-sensitive migration
contribution, using each sample's own baseline normalization. All components
retain their common-run covariance. These 18 extra functions are exploratory
diagnostics only. The migration contribution mixes recoil and azimuth
correlations and is not identified as an isolated shower error.
The 72 exploratory functions are placed in the independent reference stream,
sharing its numerical-precision budget. They remain unopened until the training
choice is frozen. The training vector has 152 entries; the reference vector has
112 entries including the new cuts and eta spectra. The first 2k-point technical
pilot put these sums in training; no production used that draft card layout.
The 72 disjoint reference sums are implemented as eight nine-bin histograms,
reducing repeated histogram work while preserving every integral and common-run
covariance. Identical-seed comparisons check the packed sums against the earlier
one-bin profile representation. Logical labels retain the same definitions.

Independent validation precision is assessed without opening central shapes.
Initial pilots determine channel allocation; all seeds are preserved and
actual unequal channel counts enter every covariance. The target is absolute
acceptance MC uncertainty <=0.0005, conditional hard acceptance uncertainty
<=0.002, and absolute charge-asymmetry uncertainty <=0.002 in every declared
bin where feasible. Report any remaining miss explicitly rather than merge
bins or tune after unblinding.

### Numerical closure implementation, before first training fits
The generic solver's convergence flag alone is insufficient. Each imposed target
must agree within one central input standard error at every coherent scale
(with an absolute1e-10 numerical floor), and every weight must be finite,
positive and normalized. A failed addition is retained as a failed candidate
and is not accepted by the combined sequence. This is a numerical target
review, not a predictive-accuracy certificate. If discarded covariance modes
require review, that flag remains explicit and the overall sequence is marked
inconclusive even when retained modes resolve an addition. No validation data
were opened to set these checks.
