"""Physical selection bookkeeping and correlated derived W predictions."""
import numpy as np

from examples.w_predictions import acceptances, charge_asymmetry, observable_groups
from maxent_upgrade import Prediction
from test_w_physical_inputs import decay_events


def test_w_ledgers_partition_the_original_sample_and_preserve_cut_identity():
    events, _, _, _ = decay_events()
    groups = observable_groups(events, exploratory=True)
    cuts, ledger = groups["cuts"], groups["old_cut_ledger"]
    values = dict(zip(ledger["labels"], ledger["values"].T))
    np.testing.assert_array_equal(cuts["values"][:, 0], 1)
    for category in ("qc", "qm"):
        for selection in ("pt0", "pt27", "pt30"):
            total = sum(values[f"expl_{category}_{q}_{r}_{selection}"] for q in range(3) for r in range(3))
            assert np.isin(total, [0, 1]).all()
            if selection == "pt0":
                np.testing.assert_array_equal(total, 1)
            else:
                pt = np.hypot(events["l_born"][:, 0], events["l_born"][:, 1])
                np.testing.assert_array_equal(total, pt > int(selection[2:]))
    for cut in (27, 30):
        physical = sum(values[f"expl_qc_{q}_{r}_pt{cut}"] for q in range(3) for r in range(3))
        projected = sum(values[f"expl_qc_{q}_{r}_proj{cut}"] for q in range(3) for r in range(3))
        # The signed migration contribution retains both positive and negative
        # events; it is not a separately renormalized event selection.
        np.testing.assert_array_equal(projected + (physical - projected), physical)
    for name in ("eta28", "eta32", "eta_mt60"):
        assert (groups[name]["values"].sum(axis=1) <= 1).all()


def raw_prediction(mean, labels, source):
    mean = np.asarray(mean, float)[None, :]
    covariance = np.eye(mean.shape[1]) + .2
    numerical = {source: dict(jacobian=np.eye(mean.shape[1]), covariance=covariance,
                              central_raw=mean[0], identity=dict(sample_id=source, run_ids=()))}
    return Prediction(mean, tuple(labels), ("central",), numerical, {})


def finite_jacobian(function, mean):
    step = 1e-5
    basis = np.eye(len(mean)) * step
    return np.column_stack([(function(mean + row) - function(mean - row)) / (2 * step) for row in basis])


def test_w_acceptance_and_asymmetry_response_matches_finite_differences():
    from examples.w_predictions import CUTS
    labels = ("norm_born",) + tuple("hold_" + k for k in CUTS)
    values = np.array([100., 70, 55, 20, 12, 65, 40])
    raw = raw_prediction(values, labels, "common-integrals")
    fractions = acceptances(raw)
    jac = finite_jacobian(lambda a: acceptances(raw_prediction(a, labels, "common-integrals")).mean[0], values)
    np.testing.assert_allclose(fractions.covariance, jac @ raw.covariance @ jac.T, rtol=2e-8, atol=1e-12)
    np.testing.assert_allclose(fractions.mean[0], [.7, .55, .2, .12, .65, .4, .6])
    plus = raw_prediction([40., 20.], ("bin0", "bin1"), "plus")
    minus = raw_prediction([30., 25.], ("bin0", "bin1"), "minus")
    asym = charge_asymmetry(plus, minus)
    np.testing.assert_allclose(asym.mean[0], [1 / 7, -1 / 9])
    # The same prediction in both slots has identically zero asymmetry and
    # variance: shared fluctuations cancel before covariance construction.
    identical = charge_asymmetry(plus, plus)
    np.testing.assert_allclose(identical.mean, 0, atol=1e-15)
    np.testing.assert_allclose(identical.covariance, 0, atol=1e-30)
