# W prior generation inputs

The public compact W fixtures use the existing 7 TeV MadGraph LO + Pythia 8.312
productions, with one million generated events per charge. These files provide
the process, run and parameter cards, and the original shower/extraction program.
The shower uses a power shower, matrix-element corrections off, primordial
hard kT of 1 GeV, hadronization on and lepton/other charged-particle QED showers
off. The hard calculation uses CT18NNLO (LHAPDF 14000); shower/MPI PDF settings
are Pythia defaults, with no override in the archived extractor.

With MadGraph, LHAPDF/CT18NNLO and the Pythia bindings already installed, a
compatible production can be generated from the repository root as follows:

```bash
mg5_aMC runcards/w_prior/wp.mg5
cp runcards/w_prior/w_run_card_wp.dat w_lo_wp/Cards/run_card.dat
cp runcards/w_prior/param_card_wp.dat w_lo_wp/Cards/param_card.dat
w_lo_wp/bin/generate_events run_01 -f
python runcards/w_prior/shower_w_v6.py \
  --lhe w_lo_wp/Events/run_01/unweighted_events.lhe.gz \
  --out my_wp_prior.npz --no-mec --kt-hard 1.0 --seed 731131
python examples/export_w_fixture.py wp my_wp_prior.npz my_wp_fixture.npz
```

Replace `wp` by `wm` for the other charge. The process card uses relative output
paths; run and parameter cards are copied from the original production. The
explicit shower seed above gives a reproducible *new* compatible sample, not
the exact original shower stream. The supplied compact fixtures retain hashes
of their actual parent files and their selection rule.

The extractor's stored `in_fid` flag uses a 25 GeV lepton cut. The study and
fixture exporter deliberately recompute the baseline at 20 GeV, with neutrino
pT>25 GeV, mT>40 GeV and |y_l|<2.5. Signal leptons are selected by hard-process
ancestry, not by taking an arbitrary leading decay lepton. The actual
`find_w_decay` implementation is authoritative where historical header prose
describes an earlier choice.

The prior has MW=80.419 GeV and width 2.0476 GeV, whereas the released
fixed-order W study uses MW=80.385 GeV and width 2.085 GeV. Production mass
moments are included in the fit; the generation inputs should not be described
as numerically identical to the fixed-order electroweak inputs.

No generator installation is required to run the committed compact examples.
