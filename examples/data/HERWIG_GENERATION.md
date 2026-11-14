# Independent Herwig Drell–Yan prior

Repeat the same public fit, uncertainty propagation and fixed-order comparisons
with the second generator:

```bash
python examples/real_drell_yan.py --prior examples/data/herwig_public_prior.npz --output herwig_demo --plot
```

The compact sample contains **98,808 events**, selected from 572,823 fiducial
events in a 1.6-million-event Herwig production. It is a separate, usable fitting
demonstration. The manuscript's generator comparison uses the full selected
samples; its event statistics must not be attributed to these public subsets.

The production used Herwig 7.3.0p1, the LCG109 build
`7.3.0p1-c7a43`, with Born `pp -> gamma*/Z -> e+e-` at 13 TeV, generation mass
50–2200 GeV and no generation-level lepton cuts. The hard-process PDF is CT18NNLO.
The standard angular-ordered QCD shower includes its default hard-emission
matrix-element correction. Shower and MPI use the default CT14lo PDFs, with
MPI and hadronization enabled. Lepton QED radiation is disabled. The production
installation was queried directly; its [recorded settings](../../runcards/herwig_dy/production_settings.txt)
confirm both PDF choices, the QCD shower, hard-emission correction and cluster
hadronization. These are two
complete generator priors with different shower/tune settings; the comparison
does not isolate a single shower algorithm parameter.

There are 160 independent shards, each with 10,000 generated events. Shard IDs
are 1..160 and seeds are `731101 + shard_id`. The [base card and production
scripts](../../runcards/herwig_dy/) record the setup. `Herwig.in` is the original
1000-event pilot template; `run_shard.sh` sets the production count, print count
and shard seed. Its `/work` paths describe the production container and can be
replaced by local paths. A local Herwig/LHAPDF installation and the optional
HepMC reader are needed only to regenerate events, not to run the supplied fit.
The script streams HepMC2 with an explicit format through the public adapter,
checks all-event counts/weight sums and optionally retains selected full records.

The fiducial selection is identical to the released DY table: 66<mll<116 GeV,
both lepton pT>27 GeV and |eta|<2.5, using prompt pre-QED electrons. The subset
rule was fixed before portability fits: seed 731131, recoil strata
`[0,30,60,100,200,500,infinity)` GeV and caps
`[50000,20000,20000,15000,12000,12000]`. Sampling is uniform without replacement
within each stratum; every event is retained when the population is below its
cap. The fit weight is `original_weight / inclusion_probability`.

The arrays preserve the original nominal weights, inclusion probabilities,
lepton four-vectors, native `generator_event_number`, `shard_id`, and globally
unique `event_id = generator_event_number + shard_id * 10000000`. The JSON retains
all shard counts, weights and individual card hashes. Retained full-record
subsets use native IDs; apply the documented shard mapping to associate them
with global IDs, or attach weights separately within each shard.

The supplied archive SHA256 is
`325ad789997dae8978e6801e43802e391aea93848dea0255604cb063ca5c67c9`.
The fixed-order moments and independent angular reference are exactly those
used by the default Pythia demonstration. No generator settings, moments, maps
or analysis cuts were tuned using its validation agreement.

## Demonstration check

[herwig_demo_check.json](herwig_demo_check.json) records a fresh central fit, six
scale members, twenty correlated integration replicas and two window variations.
The saved output preserves the input IDs, original weights and sampling
probabilities exactly; its central weights sum to 710.3611257 pb. The report
retains disagreement with the independent angular reference as a physics result.
Runtime is an observed fit-only measurement on two OpenBLAS threads, not a
performance guarantee.

The event-record adapter was also checked on a separate 1000-event Herwig pilot:
365 events passed this fiducial selection, and the first 200 selected records
were fitted with three released mass moments and written back using reversed
array order. IDs, original weights, particle momenta, statuses and vertex links
were preserved; the appended weights agreed by event ID. This checks record
association, not the precision of a full fit to 200 events. Automated small-record
tests additionally check multiple named variations, numeric/vector run metadata
and compressed output.
