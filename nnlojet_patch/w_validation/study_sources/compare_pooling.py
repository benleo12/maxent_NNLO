"""Independent raw-file versus compact-base pooling cross-check."""
from pathlib import Path
import argparse,json,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();report={}
for ch in ['WP','WM']:
 p=np.load(str(a.root/'compact_ultimate'/('V_%s.npz'%ch)),allow_pickle=False)
 q=np.load(str(a.root/'compact_raw_crosscheck'/('V_%s.npz'%ch)),allow_pickle=False)
 assert np.array_equal(p['seeds'],q['seeds']) and np.array_equal(p['n_runs'],q['n_runs'])
 row={}
 for name in ['acc','integration_errors','mean','cov']:
  x=p[name];y=q[name];assert np.array_equal(np.isnan(x),np.isnan(y))
  finite=np.isfinite(x);assert np.allclose(x[finite],y[finite],rtol=1e-12,atol=1e-13),name
  row[name+'_max_abs_difference']=float(np.max(np.abs(x[finite]-y[finite])))
 report[ch]=row
(a.root/'compact_ultimate/RAW_POOL_CROSSCHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
