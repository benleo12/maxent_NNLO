# What a saved fit records

The command-line fit and `EnsembleResult.arrays()` save a scalar string array
called `provenance_json`. Read it without enabling pickle:

```python
import json
import numpy as np

with np.load("posterior.npz", allow_pickle=False) as result:
    record = json.loads(str(result["provenance_json"]))
print(record["configuration"])
print(record["moment_table"])
print(record["normalization"], record["target_cross_section_pb"])
```

The record includes the full configuration and effective defaults, fitted moment
orders, input-array identities including IDs, transformed moment hash, raw table
identity, selected table options, normalization and central output-weight hash.
Released inputs also identify the nominal and applicable auxiliary table files
by SHA256. The command line adds the source NPZ's filename and SHA256. A hash of
the implementation modules distinguishes changed code even without a version bump.

Every uncertainty variation records its configuration, moment hash, total cross
section and normalized weight hash, plus a changed-prior fingerprint when used.
The released uncertainty workflow records requested sources, bootstrap count
and seed. Event-array fingerprints retain dtype and shape as well as the SHA256
of their contiguous bytes, so changing IDs or event ordering is detectable.

`normalization="pb"` means replacement weights sum to the stated cross section;
`"unit_area"` means they sum to one. The target cross section is retained in both
cases. The fit already incorporates the prior weights. Do not multiply the
replacement weights by the originals a second time.

For direct `upgrade` calls, use `fit_provenance(events, moments, config, result,
cross_section=...)` and serialize with `canonical_json` from
`maxent_upgrade.provenance`. The optional `table=` argument accepts a
`table_identity(...)` record for a custom table. A fingerprint identifies data;
it does not check the fiducial definition or establish physics accuracy. Keep
the original input arrays and tables alongside the record to rerun a fit.
