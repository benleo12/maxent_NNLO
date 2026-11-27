"""Born-level dilepton recoil and acoplanarity in the ATLAS fiducial region."""
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parent / "data/paper_distributions/drell_yan"


def read_bins(path, names=("low_GeV", "high_GeV")):
    """Read adjacent intervals and return their edges and named columns."""
    rows = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    low, high = (rows[k] for k in names)
    if not np.array_equal(high[:-1], low[1:]) or not np.all(high > low):
        raise ValueError(f"Noncontiguous or empty intervals in {path}")
    return np.r_[low, high[-1]], rows


def measured_shape(directory=DATA, observable="recoil"):
    if observable == "phistar":
        edges, rows = read_bins(Path(directory) / "atlas_phistar.csv", ("low", "high"))
        return edges, rows, rows["atlas"], rows["atlas_error"]
    if observable != "recoil":
        raise ValueError(f"Unknown measured observable {observable}")
    edges, rows = read_bins(Path(directory) / "atlas_recoil.csv")
    error = np.hypot(rows["data_corr_error_GeV_inv"], rows["data_uncorr_error_GeV_inv"])
    return edges, rows, rows["data_GeV_inv"], error


def phistar_reference(directory=DATA):
    """Normalized NNLO shape and its seven scale members in the four hard bins."""
    edges, rows = read_bins(Path(directory) / "phistar_fixed_order.csv", ("low", "high"))
    members = np.stack([rows[f"fixed_order_scale_{i}"] for i in range(7)])
    return dict(edges=edges, central=members[0], scale_members=members,
                sample_error=rows["fixed_order_mc_error"],
                scale_low=np.min([rows[f"fixed_order_scale_{i}_bin_low"] for i in range(7)], axis=0),
                scale_high=np.max([rows[f"fixed_order_scale_{i}_bin_high"] for i in range(7)], axis=0))
