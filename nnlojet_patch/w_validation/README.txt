Reproducing the independent W fixed-order inputs

NNLOJET v1.0.2, with the baseline moment patch in maxent_NNLO/nnlojet_patch,
is required. Its W process supplies eval_w_ptw, eval_w_ctw, eval_costhCS_w,
the Chebyshev recurrence, and the normal single-observable moment towers.
On a PRIVATE source copy, run apply_w_validation.py /path/to/nnlojet/source.
It inserts exactly the additional measurement functions archived here from
the compiled study source and adds their charge-specific bindings.

Build NNLOJET and nnlojet_w using a Fortran compiler, LHAPDF 6 with CT18NNLO,
and the normal project build instructions. The study used a Release build,
-ffree-line-length-none, and disabled OpenMP, PineAPPL, APPLfast, and DOKAN.
Full source and executable hashes are in provenance.json/build_checksums.txt.
Electroweak parameters were unchanged by this patch.

cards/WP|WM/T|V/LO|R|V|RR|RV|VV/rc.run are the ACTUAL production cards.
T contains only training moments. V contains only held-out lepton spectra,
threshold acceptances and independent diagnostic spectra. Their RUN names
and random seeds are separate. Both contain the baseline normalization.
Some inherited comment headers describe the older recoil basis; the actual
HISTOGRAMS entries and supplied measurement functions define this study.
The recoil polynomials are booked directly on log[25,500] GeV; they must not
undergo the old log[25,2500] -> log[25,500] affine transformation a second time.
Joint functions include n=0: w(c) T_k(|yW|/2-1) T_n(c), k=1..4,n=0..4.

The study reused adapted grids for exactly the same phase space. Grid sources
and hashes are recorded in provenance.json. To run elsewhere without those
grids, first perform an independent warmup, e.g. warmup=40000[5], then retain
its grid for production. Do not count warmup events as prediction samples.
Our production budget was 32 disjoint seeds per charge/group/channel and
50000[5] points per seed. Reproduction with newly adapted grids reproduces
the result statistically, rather than bit for bit.

Run each seed in its own directory, copying its channel grid and card there:
  NNLOJET --run rc.run --iseed SEED
Training seed:   710000 + charge_offset + channel_offset + replica
Validation seed: 910000 + charge_offset + channel_offset + replica
charge_offset is 0 for WP and 20000 for WM. channel_offset is 0,1000,...,5000
in the order LO,R,V,RR,RV,VV. replica=0..31 for the original sample. The independent validation RR
first precision extensions use replica=32..639 for RR, 32..255 for R, and
32..511 for RV. A final variance-based extension uses a NEW disjoint seed
formula below NNLOJET's 1000000 ceiling; see FINAL_PRECISION_PROTOCOL.txt.
Final counts (LO,R,V,RR,RV,VV) are WP=(864,2976,96,3296,768,32) and
WM=(1056,1760,64,1824,736,32). Exact physical seeds are preserved in the
per-charge validation_runs.npz arrays. Do not extend the old channel-offset
formula beyond its disjoint ranges. Pilots are excluded from every result.
All three precision protocols record the sequence of decisions made before
opening the central held-out comparisons.
The archived run_pool.py records the actual cluster orchestration; WVAL_ROOT
sets its study directory, and its library/data paths need local adjustment.

Histogram text is a density: multiply every row by its bin width, then divide
by 1000 to obtain integrated pb. Profile histograms are signed weighted sums,
not averages normalized by their own entries. Average equal-budget seeds
within each channel, then sum channels. Preserve common seed rows across all
observables and scales when resampling. aggregate.py and stat_utils.py record
the exact pooling and independent-channel bootstrap used in this study.

The published training JSON and training_runs.npz permit reproduction of all
weight fits without rerunning NNLOJET. Held-out validation caches are separate
and must not enter constraint selection. PROTOCOL.txt records the frozen rule.
