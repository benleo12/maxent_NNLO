"""Independent fixed-order units, covariance and end-to-end event weights."""
from copy import deepcopy
import json

import numpy as np
import pytest

from maxent_upgrade import MomentAccumulator, book_targets, fit_targets, merge_bookings


def card():
    profile = dict(a=1.5, b=2.5, c=6., d=8.)
    return dict(born={"mass": dict(map=["bw", 90., 2.4], range=[60, 120])},
                recoil={"pt": dict(map="log", range=[1.5, 8.], soft_lo=1.,
                                   transition="windowed", profile=profile)},
                mixed={"mass_pt": dict(observables=["mass", "pt"], n=2, profile={"pt": profile})})


def independent_reference(events):
    pt, mass = events["pt"], events["mass"]
    bw = lambda m: np.arctan((m**2 - 90.**2) / (90.*2.4))
    m = 2*(bw(np.clip(mass, 60, 120))-bw(60.))/(bw(120.)-bw(60.))-1
    p = 2*np.log(np.clip(pt, 1., 8.))/np.log(8.) - 1
    s = lambda t: t**3 * (10 - 15*t + 6*t*t)
    w = s(np.clip(np.log(pt/1.5)/np.log(2.5/1.5), 0, 1)) * (1 - s(np.clip(np.log(pt/6.)/np.log(8./6.), 0, 1)))
    t = lambda x, n: np.cos(n*np.arccos(x))
    return np.column_stack([np.ones_like(pt), t(m, 2), t(m, 4), w, w*t(p, 1), w*t(p, 3),
                            w, w*t(m, 1)*t(p, 1), w*t(m, 2)*t(p, 2)])


def records(count=96):
    rng = np.random.default_rng(123)
    # A real contribution and its correlated counterevent have different kinematics.
    p = rng.uniform(.7, 9., count)
    m = rng.uniform(84., 96., count)
    events = {"pt": np.column_stack([p, p*1.01]).ravel(),
              "mass": np.column_stack([m, m+.1]).ravel()}
    w = np.column_stack([rng.uniform(6., 7., count), -rng.uniform(4., 5., count)]).ravel()
    weights = np.stack([w, w*(1.12 + .02*events["pt"]), w*(.91 - .01*events["pt"])])
    return events, weights


def cut(events, indices):
    return {k: v[indices] for k, v in events.items()}


def test_signed_groups_partition_and_all_map_window_covariances(tmp_path):
    events, weights = records()
    n = len(weights[0])//2
    ids = np.repeat(np.arange(n), 2)
    opts = dict(scale_labels=["nominal", "up", "down"],
                degrees={"mass": [4, 2], "pt": [3, 1], "mass_pt": [[2, 2], [1, 1]]}, keep_groups=True)
    one = MomentAccumulator(card(), job_id="one", **opts)
    one.fill_groups(ids, events, weights)
    by_hand = independent_reference(events)
    vectors = np.stack([(w[:, None]*by_hand).reshape(n, 2, -1).sum(axis=1) for w in weights], axis=1)
    np.testing.assert_allclose(one.retained_groups()[1], vectors, atol=2e-14)
    assert one.raw_labels == ("fiducial", "mass:T2", "mass:T4", "pt:window", "pt:T1", "pt:T3",
                              "mass_pt:window", "mass_pt:T1,1", "mass_pt:T2,2")
    left, right = [MomentAccumulator(card(), job_id=key, **opts) for key in ("first", "second")]
    for i in range(n):
        segment = slice(2*i, 2*i + 2)
        (left if i < 37 else right).fill_group(i, cut(events, segment), weights[:, segment])
    left.save(tmp_path / "left.npz")
    right.save(tmp_path / "right.npz")
    together = merge_bookings([tmp_path / "right.npz", tmp_path / "left.npz"])
    whole = merge_bookings([one])
    raw = vectors.mean(0)
    raw_cov = np.cov(vectors[:, 0].T) / n
    np.testing.assert_allclose(together.raw_mean, raw, atol=1e-14)
    np.testing.assert_allclose(together.raw_covariance, raw_cov, atol=1e-14)
    np.testing.assert_allclose(together.target_covariance, whole.target_covariance, atol=1e-14)
    # Reference derivatives via finite differences of *all* normalized inputs.
    def normalize(v):
        return np.r_[v[0], v[[1, 2]]/v[0], v[3]/v[0], v[[4, 5]]/v[3], v[6]/v[0], v[[7, 8]]/v[6]]
    step = np.eye(raw.shape[1])*1e-5
    jac = np.column_stack([(normalize(raw[0]+d)-normalize(raw[0]-d))/(2e-5) for d in step])
    np.testing.assert_allclose(together.target_covariance, jac @ raw_cov @ jac.T, atol=1e-11)
    assert abs(together.target_covariance[0, 3]) > 1e-5
    direct = together.targets(replicas=5, seed=91)
    other = whole.targets(replicas=5, seed=91)
    assert direct["integration"]["stat_method"].startswith("bootstrap")
    assert [r["cross_section_pb"] for r in direct["variations"]["scale"]] == pytest.approx(raw[1:, 0])
    assert [r["cross_section_pb"] for r in direct["variations"]["stat"]] == pytest.approx(
        [r["cross_section_pb"] for r in other["variations"]["stat"]])
    np.testing.assert_allclose([r["value"] for r in direct["moments"]],
                               [r["value"] for r in other["moments"]], atol=1e-14)


def test_independent_channels_signed_rates_run_ids_and_normalization(tmp_path):
    config = {"born": {"x": {"range": [-1, 1], "map": "lin"}}}
    a = np.array([[10, 1], [15, 1.1], [9, .8], [14, 1.4]], dtype=float)
    v = np.array([[-2, -.2], [-3, -.1], [-1, -.3]], dtype=float)
    books = []
    for label, rows in (("real", a), ("virtual", v)):
        book = MomentAccumulator(config, job_id=label+"-stream", channel=label, unit="run", orders=1,
                                 keep_groups=True)
        for i, row in enumerate(rows):
            book.add_integrals(i, row)
        books.append(book)
    merged = merge_bookings(books)
    raw = a.mean(0) + v.mean(0)
    cov = np.cov(a.T)/len(a) + np.cov(v.T)/len(v)
    np.testing.assert_allclose(merged.raw_mean[0], raw)
    np.testing.assert_allclose(merged.raw_covariance, cov)
    table = merged.targets()
    assert table["cross_section_pb"] == raw[0]
    assert table["moments"][0]["value"] == pytest.approx(raw[1] / raw[0])
    assert table["integration"]["channel_counts"] == {"real": 4, "virtual": 3}
    with pytest.raises(ValueError, match="duplicate job ID"):
        merge_bookings([books[0], books[0]])
    with pytest.raises(ValueError, match="at least two independent"):
        b = MomentAccumulator(config, job_id="single", orders=1)
        b.add_integrals(0, [1, .1])
        merge_bookings([b])
    path = tmp_path / "saved.npz"
    books[0].save(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        books[0].save(path)
    assert path.read_bytes() == before
    loaded = MomentAccumulator.load(path)
    ids, values = loaded.retained_groups()
    np.testing.assert_array_equal(ids, np.arange(4))
    np.testing.assert_array_equal(values[:, 0], a)
    with pytest.raises(ValueError, match="group IDs"):
        loaded.add_integrals(3, [10, .5])
    assert loaded.count == 4


def test_zero_trials_and_compare_ungrouped_signed_error():
    config = {"born": {"x": {"range": [-1, 1], "map": "lin"}}}
    x = np.linspace(-.6, .6, 100)
    book = MomentAccumulator(config, job_id="signed", orders=1)
    # Nearly complete cancellation on each draw. Treating the pair as separate
    # trials would grossly overestimate the error on the normalized moment.
    weights = np.tile([101., -100.], len(x))
    data = np.repeat(x, 2)
    book.fill_groups(np.repeat(np.arange(len(x)), 2), {"x": data}, weights)
    book.fill_group(150, {}, [])  # rejected trial; gaps in identifiers do not count
    result = merge_bookings([book])
    assert result.raw_mean[0, 0] == pytest.approx(100/101)
    grouped_error = result.targets()["moments"][0]["stat_error"]
    naive = book_targets({"x": data, "weight": weights}, config, orders=1)["moments"][0]["stat_error"]
    assert naive > 100 * grouped_error
    replica = result.targets(replicas=20, seed=4)
    assert replica["integration"]["stat_method"].startswith("Gaussian")
    assert replica == result.targets(replicas=20, seed=4)
    with pytest.raises(ValueError, match="keep_groups"):
        book.retained_groups()
    changed = deepcopy(config)
    changed["born"]["x"]["range"] = [-2, 2]
    other = MomentAccumulator(changed, job_id="new", orders=1)
    other.add_integrals(0, [4, 0])
    with pytest.raises(ValueError, match="agree on maps"):
        merge_bookings([book, other])


@pytest.mark.parametrize("transition", ["windowed", "blend"])
def test_correlated_window_error_in_actual_positive_event_fit(transition):
    config = {"recoil": {"pt": {"map": "log", "range": [1., 5.], "profile": {"a": 1., "b": 2., "c": 5.},
                                "transition": transition}}, "moment_selection": False}
    z = np.linspace(-1, 1, 61)
    # Raw fiducial integral, window integral, first window numerator.
    v = np.column_stack((12 + .4*z, 6 + .8*z, -.6 + 1.2*z + .1*z*z))
    b = MomentAccumulator(config, job_id="runs", unit="run", orders=1, keep_groups=True, scale_labels=("nom", "up"))
    for i, row in enumerate(v):
        b.add_integrals(i, np.stack([row, row * [1.2, 1.15, 1.12]]))
    result = merge_bookings([b])
    targets = result.targets(replicas=3, seed=27)
    pt = np.linspace(.1, 5.03, 900)
    prior = {"pt": pt, "weight": np.ones(len(pt))}
    fit = fit_targets(prior, targets)
    row = fit.constraint_checks["central"][1]
    raw = result.raw_mean[0]
    r, m = raw[1] / raw[0], raw[2] / raw[1]
    a = 0. if transition == "windowed" else (row["target"] - r*m)/(1-r)
    jac = np.array([(-raw[2]+a*raw[1])/raw[0]**2, -a/raw[0], 1/raw[0]])
    sigma = np.sqrt(jac @ result.raw_covariance @ jac)
    np.testing.assert_allclose(row["integration_error"], sigma, rtol=2e-12)
    scaled = result.raw_mean
    scale_values = scaled[:, 2] / scaled[:, 0] + (1-scaled[:, 1]/scaled[:, 0])*a
    np.testing.assert_allclose(row["input_error"], np.hypot(sigma, np.ptp(scale_values)/2), rtol=2e-12)
    assert fit.arrays()["weight_stat"].shape == (3, len(pt))
    # The same output sample supports a fresh observable without another solve.
    select = pt > 2.3
    for i, member in enumerate(targets["variations"]["stat"]):
        ws = fit.arrays()["weight_stat"][i]
        assert np.isclose(ws.sum(), member["cross_section_pb"])
        assert 0 < ws[select].sum() < ws.sum()
    assert json.loads(str(fit.arrays()["constraint_checks_json"]))["central"][1]["input_error_model"].startswith("moment-rate")


def test_a_signed_job_stores_only_sufficient_statistics_by_default(tmp_path):
    cfg = {"recoil": {"pt": {"range": [1., 5.], "map": "log"}}}
    booking = MomentAccumulator(cfg, job_id="seed-42", degrees={"pt": [2, 4]})
    assert booking.raw_labels == ("fiducial", "pt:window", "pt:T2", "pt:T4")
    rng = np.random.default_rng(42)
    pt = rng.uniform(0, 7, 5000)
    w = .5 + rng.random(len(pt))
    booking.fill_groups(np.arange(len(pt)), {"pt": pt}, w)
    path = tmp_path / "compact.npz"
    booking.save(path)
    with np.load(path) as saved:
        assert set(saved.files) == {"mean", "scatter", "metadata"}
        assert saved["scatter"].shape == (1, 4, 4)
    merged = merge_bookings([path])
    draws = merged.targets(replicas=1800, seed=5)["variations"]["stat"]
    reconstructed = np.array([[v["cross_section_pb"], v["rates"]["pt"]["value"]*v["cross_section_pb"],
                               *[q["value"]*v["rates"]["pt"]["value"]*v["cross_section_pb"] for q in v["moments"]]]
                              for v in draws])
    assert np.all(np.abs(reconstructed.mean(0) - merged.raw_mean[0]) < 0.1*np.sqrt(merged.raw_covariance.diagonal()))
    sd = np.sqrt(merged.raw_covariance.diagonal())
    np.testing.assert_allclose((np.cov(reconstructed.T) - merged.raw_covariance) / sd[:, None] / sd[None, :],
                               np.zeros((4, 4)), atol=.1)


def test_reject_mismatched_units_scales_and_nonindependent_identities(tmp_path):
    config = {"born": {"x": {"range": [0, 2], "map": "lin"}}}
    a = MomentAccumulator(config, unit="group", job_id="a", orders=1)
    a.fill_group(0, {"x": [1.]}, [2.])
    with pytest.raises(ValueError, match="ID"):
        a.fill_groups([1, 0], {"x": [.5, .8]}, [1., 1.])
    with pytest.raises(ValueError, match="ID"):
        a.fill_group(-10, {"x": [.5]}, [1.])
    with pytest.raises(ValueError, match="finite"):
        a.fill_group(1, {"x": [np.nan]}, [1.])
    assert a.count == 1
    for settings in ({"unit": "run"}, {"scale_labels": ("nominal", "up")}):
        b = MomentAccumulator(config, job_id="b", orders=1, **settings)
        b.add_integrals(7, np.ones((len(b.scale_labels), 2)))
        with pytest.raises(ValueError, match="agree on maps"):
            merge_bookings([a, b])
    b = MomentAccumulator(config, job_id="b", orders=1, keep_groups=True)
    for i in range(3):
        b.add_integrals(i, [2, .1+i/10])
    path = tmp_path / "retained.npz"
    b.save(path)
    with np.load(path) as f:
        bad = dict(f)
    bad["group_integrals"][2, 0, 0] *= 3
    replacement = tmp_path / "stale.npz"
    np.savez(replacement, **bad)
    with pytest.raises(ValueError, match="disagree"):
        merge_bookings([replacement])
