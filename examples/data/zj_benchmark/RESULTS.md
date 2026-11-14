The frozen comparison demonstrates useful transfer from a finite set of moments
to jet selections and balance distributions that were never fitted. It also
shows where increasing the joint degree stops producing resolved changes in the
chosen analysis selections. It does not establish a universal automatic selector
or NNLO+PS accuracy.

The primary jet results are valid. The optional angular reference has also been
corrected and validated, as documented in TECHNICAL_CORRECTION.txt. Its
normalization integrals exactly match all original validation runs and scales;
all primary arrays are unchanged. The corrected angular discrepancy is 30.25%
for the prior and 32.03% for J=4. The two jet magnitudes therefore do not repair
the missing angular information. This diagnostic did not enter any fitting or
model-selection decision.

The study used 16,004 equal-weight prior events and exactly the same 16 million
training integration points for every method. A further 16 million independent
points supplied the validation reference. The 19 held-out cells with at least 50
prior events contain 97.021% of its probability. The table uses half the summed
absolute probability differences on those cells, without renormalizing them.

| Method | Shape constraints | 2D discrepancy | Balance discrepancy,250–500 GeV | Effective events |
|---|---:|---:|---:|---:|
| Prior |0|10.475%|28.072%|16,004|
| Separate moments,D=4 |8|1.716%|8.916%|12,962|
| Separate D=4 + joint J=1 |9|1.845%|4.433%|12,840|
| Separate D=4 + joint J=2 |12|1.473%|3.311%|12,556|
| Separate D=4 + joint J=3 |17|1.247%|2.389%|12,518|
| Separate D=4 + joint J=4 |24|1.165%|2.165%|12,310|
| Marginal histogram IPF,B=5 |8|4.441%|10.135%|13,652|
| Joint histogram ratio,B=5 |14|4.348%|9.703%|13,644|

Without the population mask, the full-grid discrepancies are 11.50% for the
prior, 1.925% for separate moments, 1.762% for J=2, 1.528% for J=4 and
4.756% for the B=5 joint histogram. The ordering is therefore not caused by
the population restriction.

The direct comparison at 12 versus 14 constraints is useful: J=2 gives 1.473%
against 4.348% for the B=5 joint histogram at the same fixed-order point budget.
The respective numerical spreads of the 2D distances are 0.313 and0.195 percentage
points. These are bootstrap spreads, not theory uncertainties or evidence that
this representation is optimal for arbitrary processes. The histogram number
uses only the 19 successful support-preserving replicas out of 32; this conditional
error estimate must be identified wherever it is shown.

The three independent cut efficiencies are 0.362087±0.000399,
0.141245±0.000190 and0.0330424±0.0000555, where the errors are integration
statistics. Relative errors of the prior are −24.97%,−43.95%,−62.37%.
Separate D=4 moments give +0.10%,+2.52%,+6.92%. J=2 gives −0.12%,+0.69%,−2.20%;
J=4 gives −0.05%,+0.33%,−3.92%. The J=4 fit-statistical uncertainties, expressed
relative to the reference efficiencies, are 0.54%,1.07%,4.57%. The hardest cut
therefore supports agreement at the few-percent numerical precision available,
not a sub-percent accuracy claim.

The first added mixed moment differs from the marginal fit by 0.001813±0.000271
under paired numerical resampling. Adding J=1 changes the three cuts by
−1.179±0.191,−2.892±0.448,−5.893±1.015 percentage points relative to the
reference efficiencies. Enlarging to J=2 gives+0.953±0.188,+1.063±0.535,
−3.228±1.134. These changes are resolved. Changes from J=2 to J=3 and from
J=3 to J=4 are unresolved at this numerical precision for all three cuts.
That provides an operational stability demonstration without selecting the
reported model retrospectively. The conditional high-pT balance distribution
still exhibits a gradual central improvement along the fixed sequence; the
correlated errors and every resolution remain available in the cache.

All moment fits succeeded in all 32 paired bootstrap replicas. Exact histogram
matching has explicit support limitations: B=7 misses 0.03763% target probability
in the highest subleading-jet bin; the B=10 joint grid misses 0.13561% in eight
cells. These tiny masses should not be described as large physics errors.
Even B=4 and B=5 can lose their rare supporting prior events in a Poisson
replica: success counts are 22/32 for B=4 IPF/ratio, 22/32 for B=5 IPF and 19/32
for B=5 ratio. Full-grid cell counts, effective counts and maximum factors are
in local_support.csv. Fine-grid failures cannot be hidden when quoting an
ordinary-comparator error bar.

Numerical constraint closure is separately audited: the largest L2-induced
retained-moment residual is 0.456 times its training fixed-order statistical
error (J=4); the others are below 0.182. Thus agreement of the imposed moments
is not being mistaken for the independent cut/balance checks. The J=4 maximum
event-weight factor is 28.43, with 76.92% of the prior effective count retained.

The complete 21-model, 7-scale, 32-replica fit/analysis pass took 2.77 s on the mini
with at most 2 BLAS threads. This timing is for the 16,004-event hard-region
sample, not the full generated prior. The two integrations cost 439.73 core-minutes
in total (about 28 min wall time on 16 cores). The optional diagnostic correction
is recorded separately. All targets were booked simultaneously, so this does
not measure a relative integration-runtime advantage of one representation.

Scientific interpretation: the hard two-jet prior substantially underpopulates
the harder asymmetric selections. Correcting its two spectra removes most of
that effect. The mixed moments then improve how the jet momentum is shared,
which is visible in the conditional balance distribution and in the first two
resolved changes of the asymmetric efficiencies. Smooth moment constraints
also correct variations within bins that a coarse direct histogram ratio
inherits from its prior. This is evidence for useful finite-resolution
compression and interpolation; IPF is itself a KL projection, so it is not a
demonstration that entropy minimization is uniquely superior to reweighting.

The paper figure is benchmark_paper.pdf: 430.2 pt wide, 8 pt text, 7.5 pt legend,
red reserved for moment fits, legends outside axes, embedded fonts, shared
style audit clean. benchmark_overview.pdf contains the larger public overview
including both conditional balance distributions. All resolutions and full
statistical/scale arrays are retained in the compact caches.
