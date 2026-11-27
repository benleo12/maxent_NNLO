"""Recover the original LO+PS charge rates from the full-sample histograms."""
import json
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parent / "data"
TABLE = DATA / "paper_distributions/w/lo_ps_prior.csv"


def prior_bins(archive, charge):
    """The stored P histogram has the original bin fractions and the fitted total rate."""
    meta = json.loads(str(archive["metadata"]))
    shapes = archive["P_mean"]
    source = json.loads((DATA / f"w_predictivity/w_{charge}_prior.json").read_text())
    rate = source["source_selected_cross_section_pb"]
    total = shapes[:, meta["labels"].index("norm_born")]
    fractions = shapes / total[:, None]
    if not np.allclose(fractions, fractions[0], rtol=1e-10, atol=1e-13):
        raise ValueError("The archived P histograms have different shapes at different scales")
    return dict(zip(meta["labels"], rate * fractions[0]))


def prior_asymmetry(cut, edges):
    """Original generator rates at the published lepton cut, folded in absolute eta."""
    rows = np.genfromtxt(TABLE, delimiter=",", names=True)
    rows = rows[rows["cut_GeV"] == cut]
    np.testing.assert_array_equal(edges, np.r_[rows["eta_low"], rows["eta_high"][-1]])
    return rows["asymmetry"]
