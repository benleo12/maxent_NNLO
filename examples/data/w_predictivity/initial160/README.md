# Initial common-vector W training input

`T_WP.npz` and `T_WM.npz` preserve the first, predeclared training budget:
160 independent 100,000-point runs in each of the six additive NNLOJET channels,
separately for each W charge. Each file contains the complete 152-integral vector
at seven coherent scales, its central covariance, per-seed accumulators,
integration errors, physical seed identities and source records. The order is
the previous 73 production/angular integrals followed by the 79 declared new
physical functions in `../feature_definitions.json`.

The two charges use 1,920 distinct physical seeds in total. They are separate
from the validation stream. `mean` has shape `(152,7)`, `cov` is `(152,152)`,
`acc` is `(6,160,152,7)` and `seeds` is `(6,160)`. The covariance is the sum of
the six independent channel covariance-of-mean estimates. It is a covariance
of raw integrals in pb; the public interface applies the shared-normalization
Jacobian when constructing moment targets.

The compact event demonstration has its own selection and freeze identity.
It uses this fixed input budget even if the full study later extends training;
it must not silently inherit a full-sample selection or a different input file.

SHA256:

- W+: `1784b18a0d750b1a4c6fdab163f9af1080dcfe8a6bda33943f9843923de3cf96`
- W−: `e4336115c56a70cbc5a303d4fa3461332a27129aa59dd41bb39d8e38314cb430`
