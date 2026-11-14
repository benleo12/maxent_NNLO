"""Comparator closure and explicit failures in the portable benchmark."""
import json
from pathlib import Path

import numpy as np
import pytest

from examples import zj_benchmark as benchmark


def toy_problem(monkeypatch):
    definitions = json.loads((Path(__file__).resolve().parents[1] / "examples/data/zj_benchmark/definitions.json").read_text())
    monkeypatch.setattr(benchmark, "DEFS", definitions)
    edges = np.geomspace(60, 1000, 41)
    centers = np.sqrt(edges[:-1]*edges[1:])
    x, y = np.meshgrid(centers, centers)
    selected = x >= y
    x, y = x[selected], y[selected]
    u, v = 2*np.log(x/60)/np.log(1000/60)-1, 2*np.log(y/60)/np.log(1000/60)-1
    events = dict(ptj1=x, ptj2=y, weight=np.ones(len(x)), aco=np.zeros(len(x)))
    q = np.exp(.25*u-.1*v+.6*u*v)
    q /= q.sum()
    cx, cy = (np.polynomial.chebyshev.chebvander(a, 4) for a in (u, v))
    moments = np.einsum('i,ij,ik->jk', q, cx, cy)
    target = dict(norm=np.ones((1, 7)), moments=np.repeat(moments[:, :, None], 7, axis=2))
    for number, bins in definitions['train_edges'].items():
        hist = np.histogram2d(x, y, bins=bins, weights=q)[0]
        target['joint_'+number] = np.repeat(hist[:, :, None], 7, axis=2)
        target['train_x_'+number] = np.repeat(hist.sum(axis=1)[:, None], 7, axis=1)
        target['train_y_'+number] = np.repeat(hist.sum(axis=0)[:, None], 7, axis=1)
    return events, target, q


def test_comparators_close_their_own_constraints(monkeypatch):
    events, target, truth = toy_problem(monkeypatch)
    weights, report = benchmark.fit(('D1_J1', 'maxent', 1, 1), events, target)
    assert report['closure_abs'] < 1e-4
    np.testing.assert_allclose(weights, truth, rtol=4e-4, atol=1e-7)
    edges = benchmark.DEFS['train_edges']['5']
    for method in ('ipf', 'ratio'):
        weights, report = benchmark.fit((method+'_B5', method, 5, 0), events, target)
        hist = np.histogram2d(events['ptj1'], events['ptj2'], bins=edges, weights=weights)[0]
        expected = target['joint_5'][:, :, 0]
        assert weights.sum() == pytest.approx(1)
        assert report['closure_abs'] < 1e-9
        if method == 'ratio':
            np.testing.assert_allclose(hist, expected, atol=1e-12)
        else:
            np.testing.assert_allclose(hist.sum(axis=0), expected.sum(axis=0), atol=1e-9)
            np.testing.assert_allclose(hist.sum(axis=1), expected.sum(axis=1), atol=1e-9)


def test_exact_histogram_failure_reports_missing_probability(monkeypatch):
    events, target, _ = toy_problem(monkeypatch)
    edge = benchmark.DEFS['train_edges']['7'][1]
    keep = ~((events['ptj1'] < edge) & (events['ptj2'] < edge))
    incomplete = {key: value[keep] for key, value in events.items()}
    with pytest.raises(RuntimeError, match='target mass='):
        benchmark.fit(('ratio_B7', 'ratio', 7, 0), incomplete, target)


def test_benchmark_does_not_drop_negative_selected_weights(tmp_path):
    path = tmp_path / "signed.npz"
    np.savez(path, weight=[1., -1.], ptj1=[100., 90.], ptj2=[80., 70.],
             njet=[2, 2], pimdphi=[.2, .3])
    with pytest.raises(ValueError, match="strictly positive"):
        benchmark.load_prior(path)
