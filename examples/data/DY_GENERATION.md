# Drell–Yan fixture and reference recipe

The supplied `dy_prior.npz` is a compact subset of the paper's current
`dy_prior_v6_nomec10M_atlas.npz`. Its generator settings were checked against the
production cards and shower driver on 13 November 2026. The event archive itself
does not embed every generator version/build detail; the source-file hash and
unchanged event IDs identify the exact numerical input to this example.

## Prior production

The hard process was generated with MG5_aMC 3.5.3 and the `sm` model:

```text
import model sm
define p = g u c d s u~ c~ d~ s~
define l+ = e+ mu+
define l- = e- mu-
generate p p > l+ l-
output dy_lo
```

The original hard-process run card is [dy_prior_mg5.run](../../runcards/dy_prior_mg5.run).
It uses 13 TeV collisions, `50 < mll < 2200` GeV and no generation-level lepton
cuts. The original electroweak parameters are the MG5 Standard Model defaults;
the NNLO input uses the separate values in its run card. The mass constraints
can therefore correct the prior's line shape as well as its QCD shape.

The LHE events were showered with Pythia 8, with these non-default choices:

```text
Beams:frameType = 4
Beams:setProductionScalesFromLHEF = off
SpaceShower:pTmaxMatch = 2
TimeShower:pTmaxMatch = 2
BeamRemnants:primordialKThard = 1.0
SpaceShower:MEcorrections = off
TimeShower:MEcorrections = off
TimeShower:QEDshowerByL = off
TimeShower:QEDshowerByOther = off
SpaceShower:QEDshowerByL = off
```

The remaining shower and MPI settings use Pythia's defaults (Monash tune,
NNPDF2.3 QCD+QED LO shower/MPI PDF, ISR/FSR alpha_s(mZ)=0.1365). Hadronization and
MPI are enabled. The production driver temporarily switches hadronization off
to store parton-level jets, then calls hadronization and stores hadron-level jets.
Failed hadronization events are counted and omitted. Signal leptons are the
final copies of hard outgoing leptons, before QED radiation; heavy-flavour decay
leptons are not substituted for them.

Ten one-million-event hard-process runs were showered in halves of 500,000
events. The shower seeds for run `r=1..10`, half `c=0,1` were `1000+20*r+c`.
The final DY selection is `66 < mll < 116` GeV, both lepton `pT > 27` GeV and
`|eta| < 2.5`. The full selected archive contains 3,583,332 events.

## Compact public subset

`build_dy_fixture.py` fixes the sampling rule before fitting: seed 13112026,
recoil strata `[0,30,60,100,200,500,infinity)` GeV, with maximum retained counts
`[50000,20000,20000,15000,12000,12000]`. Events are sampled uniformly without
replacement inside each stratum. This produces 112,978 events and a 6.80 MB
compressed file. It deliberately retains more of the recoil tail.

`event_id` is the unchanged source `gidx`; `source_index` is its row in the full
selected archive. `original_weight` preserves the generation weight,
`inclusion_probability` records the sampling probability, and the fit uses
`weight = original_weight / inclusion_probability`. Omitting this correction
would change the prior. The JSON records the exact source SHA256 and population
of every stratum. The fixture has not been selected for favorable fit results.

The stored `ptj1` and `jet_count` describe hadron-level anti-kT R=0.4 jets above
20 GeV, with |y|<4.4 and lepton–jet separation>0.4. These extra observables can be
histogrammed with the fitted weights; no NNLO jet accuracy follows from doing so.

## Independent angle and constrained recoil references

`dy_reference.npz` contains 64 independent NNLO integrations of the signed
Collins–Soper angle, folded to its absolute value. The [run card](../../runcards/dy_costh_cs.run)
uses NNLOJET v1.0.2, CT18NNLO, the Gmu scheme, the same 13 TeV fiducial selection
and central scales muR=muF=mll, with seven scale choices. The channel sums include
LO, V, R, VV, RV and RR. These runs are independent of the moment production and
the angle is not constrained in the example. Raw densities in fb per unit angle
are converted to pb per unit angle. The reference integration error is retained,
including the regions with large cancellations. The plot shades the fixed-order
shoulder `0.75 < |cos(theta_CS)| < 0.9` and omits ratios there, as in the manuscript;
the full reference arrays and errors remain in the saved data.

The recoil reference uses 300 runs from the same production as the moment
constraints, so it tests constraint closure rather than independent physics
validation. It uses exact native bin boundaries, coarsened above 60 GeV where the
transition has finished; densities are in pb/GeV. The moment values themselves
are still the released unbinned moments, not moments reconstructed from this
histogram.

`build_dy_reference.py` records the raw-file hashes, seeds, channel combinations
and unit conversion. The compact input arrays and their JSON provenance are
committed. Running the public example needs neither a generator nor NNLOJET nor
access to the original large archives.
