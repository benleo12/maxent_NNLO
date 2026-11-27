"""Born-level ATLAS dilepton recoil and full-sample comparisons in common bins."""
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parent / "data/paper_distributions/drell_yan"


def read_bins(path):
    """Read adjacent native intervals and return their edges and named columns."""
    rows = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    if not np.array_equal(rows["high_GeV"][:-1], rows["low_GeV"][1:]):
        raise ValueError(f"Noncontiguous intervals in {path}")
    edges = np.r_[rows["low_GeV"], rows["high_GeV"][-1]]
    return edges, rows


def measured_shape(directory=DATA):
    edges, rows = read_bins(Path(directory) / "atlas_recoil.csv")
    density = rows["data_GeV_inv"]
    error = np.hypot(rows["data_corr_error_GeV_inv"], rows["data_uncorr_error_GeV_inv"])
    return edges, rows, density, error
