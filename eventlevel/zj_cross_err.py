import os, sys
os.environ["ZJ_CH"] = "LO,R,V"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zj_mixed_scan as Z
from nnlojet_moments import common_seeds, _moment_over_seeds, _reduce
NMIX = 4
seeds = common_seeds(Z.ZDIR, Z.RUN, Z.CH,
                     tag=["prof_wj12_0"] + [f"prof_wj12_{m}{n}" for m in range(1, 5)
                          for n in range(1, 5)], prefix=Z.PREFIX)
print("(m,n)  FO       sigma_FO")
_rows = []
for m in range(1, 5):
    for n in range(1, 5):
        r = _moment_over_seeds(Z.ZDIR, Z.RUN, Z.CH, seeds, f"prof_wj12_{m}{n}",
                               "prof_wj12_0", Z.PREFIX)
        c, st, sc, tot = _reduce(r, 0)
        print(f"({m},{n})  {c:+.4f}  {tot:.4f}")
        _rows.append((m, n, c, tot))
np.savez(os.path.join(HERE, "zj_cross_err.npz"), mn=np.array([(m, n) for m, n, _, _ in _rows]),
         fo=np.array([c for _, _, c, _ in _rows]), sigma=np.array([t for _, _, _, t in _rows]))
print("saved zj_cross_err.npz")
