# Independent-reference precision stopping record

Recorded after the initial64 seeds/channel precision-only audit and after the
training freeze, before opening any reference central prediction.

The original absolute precision goals remain unchanged:0.0005 for baseline
acceptances,0.002 for the hard-recoil conditional acceptance and0.002 for each
prespecified charge-asymmetry bin. No central-value agreement enters allocation.
Every original physical seed is retained with equal weight within its channel.
All reference runs have250000 integration points.

The initial audit is dominated by RR (about96% of WP pT>28 variance and95% of
the worst eta28 asymmetry-bin variance). Initial reference errors range up to
0.00335 for an acceptance and0.0146 for an asymmetry bin. These are integration
precision estimates, not statements about model agreement.

First extension: total runs per charge are LO128,R512,V128,RR1024,RV256,VV64.
The3456 additional physical runs are split by charge into independent8-node
allocations, each128 physical single-thread workers/node. Measured per-run
median times are LO7.8s,R19.1s,V67.5s,RR863.9s,RV791.6s,VV613.3s.
Expected walltime27–29min with heavy channels first. Cumulative reference
budget after this extension is1.056 billion points.

At most one further precision-only extension is permitted. Its counts must
stay within these per-charge caps: LO256,R2048,V256,RR2048,RV1024,VV128.
The absolute total cap is11520 equal-budget runs or2.88 billion reference
points, including every initial run. Allocate the final extension by measured
channel variance/cost, without exceeding these caps. The reference remains
blinded during this review. If the declared errors are still missed at the
finite stop, report the actual errors and named bins, do not merge bins or
extend again. No fitting/constraint choices can change.
