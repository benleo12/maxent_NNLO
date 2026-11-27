"""Independent signed integrals, native file parsing, and restart behavior."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from maxent_upgrade.nnlojet import _measurements, calculate


CARD = """PROCESS H
END_PROCESS
RUN HIGGS
  PDF = TestSet[0]
END_RUN
HISTOGRAMS
  ! earlier mxrow0 output is just a comment
END_HISTOGRAMS
SCALES
  muf=125 mur=125
  muf=250 mur=250
END_SCALES
CHANNELS region=all
  LO V
END_CHANNELS
"""

# A native-format integrator with analytic signed central and varied cross sections.
# The staged "grid" adds a known shift to the moment numerator in each job.
FAKE = '''#!/usr/bin/env python3
from pathlib import Path
import os
import re
import sys

base = Path(os.environ["TEST_NNLOJET"])
seed = int(sys.argv[sys.argv.index("--iseed")+1])
card = Path("moments.run").read_text()
channel = re.search(r"^CHANNELS[^\\n]*\\n\\s*(\\w+)", card, re.M)[1]
with (base / "calls").open("a") as stream:
    stream.write(f"{channel} {seed}\\n")
if (base / f"fail-{seed}").exists():
    sys.exit(1)
grid = int(Path("inputs/grid").read_text())
assert Path("maxent_measures.dat").read_text().splitlines() == ["1", "1", "abs_yh 1 1 0 4 0 0 0 0 0 0"]
tot = (1000 + 10 * seed) if channel == "LO" else -20 * seed
num = (200 + 2 * seed) if channel == "LO" else -3 * seed
bad = (base / "defect").read_text() if (base / "defect").exists() else ""
single = re.search(r"^HISTOGRAMS\\s*>\\s*(\\w+)", card, re.M)
for i, (central, varied) in enumerate([(tot, 2*tot), (num + grid, num + grid)]):
    path = Path(f"H.X.{channel}.mxrow{i}.s{seed}.dat")
    labels = "tot_scale01[1] tot_scale01_Err[2] tot_scale02[3] tot_scale02_Err[4]"
    content = f"#labels: {labels}\\n#neval: 250\\n{central} 1 {varied} 1\\n#nx: 0\\n"
    if i == 1:
        if bad == "missing":
            continue
        if bad == "ambiguous":
            Path("other."+str(path)).write_text(content)
        if bad == "nx":
            content = content.replace("#nx: 0", "#nx: 1")
        if bad == "double":
            content += "#nx: 0\\n"
        if bad == "evaluations":
            content = content.replace("250", "2", 1)
        if bad == "columns":
            content = content.replace("scale02", "scale03")
        if bad == "variance":
            content = content.replace(f"{central} 1 ", f"{central} -1 ")
        if bad == "finite":
            content = content.replace(f"{central} 1 ", "nan 1 ")
    if single:
        with Path(f"H.X.{channel}.{single[1]}.s{seed}.dat").open("a") as stream:
            stream.write(f"\\n#name: mxrow{i}\\n" + content)
    else:
        path.write_text(content)
        if bad == "regions":
            Path(str(path).replace(f".{channel}.", f".{channel}a.")).write_text(content)
if bad != "incomplete":
    print(" Elapsed time = 0.01 seconds")
'''


@pytest.fixture
def example(tmp_path, monkeypatch):
    binary = tmp_path / "bin" / "NNLOJET"
    binary.parent.mkdir()
    binary.write_text(FAKE)
    binary.chmod(0o755)
    (tmp_path / "h.run").write_text(CARD)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/grid").write_text("5")
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libnnlojet_h.so").write_bytes(b"matrix elements")
    pdf = tmp_path / "pdfs" / "TestSet"
    pdf.mkdir(parents=True)
    (pdf / "TestSet.info").write_text("metadata")
    (pdf / "TestSet_0000.dat").write_text("parton densities")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("LHAPDF_DATA_PATH", str(pdf.parent))
    monkeypatch.setenv("TEST_NNLOJET", str(tmp_path))
    setup = dict(runcard="h.run", config={"born": {"y": {"map": "lin", "range": [0, 4]}}},
                 axes={"y": "abs_yh"}, orders=1, seeds=[10, 11], scale_labels=["central", "up"],
                 dependencies=["inputs/grid"], selection="inclusive Higgs")

    def run(config=None, **kwargs):
        return calculate(tmp_path / "setup.json", setup if config is None else config,
                         binary=binary, directory=tmp_path / "work", **kwargs)
    return setup, run


def calls(base):
    return sorted((base / "calls").read_text().splitlines())


def test_signed_scales_restart_and_replica_reuse(example, tmp_path):
    _, run = example
    path = run(workers=3, replicas=4)
    table = json.loads(path.read_text())
    # Native fb. Independent LO seeds 10,11 and negative V seeds 12,13.
    np.testing.assert_allclose(table["integration"]["raw_mean_pb"], [[.855, .1935], [1.71, .1935]])
    assert table["moments"][0]["value"] == pytest.approx(.1935 / .855)
    assert table["variations"]["scale"][0]["moments"][0]["value"] == pytest.approx(.1935 / 1.71)
    assert table["integration"]["channel_counts"] == {"LO": 2, "V": 2}
    # Covariance of the mean of two independent runs in each channel, in pb^2.
    expected = (np.outer([10, 2], [10, 2]) + np.outer([-20, -3], [-20, -3])) / 4e6
    np.testing.assert_allclose(table["integration"]["raw_covariance_pb2"], expected)
    assert len(table["variations"]["stat"]) == 4
    before = calls(tmp_path)
    assert before == ["LO 10", "LO 11", "V 12", "V 13"]
    assert run(workers=1, replicas=4) == path
    assert run(replicas=6, seed=2).parent == path.parent
    assert calls(tmp_path) == before


def test_concurrent_invocations_and_corrupt_booking(example, tmp_path):
    _, run = example
    with ThreadPoolExecutor(2) as pool:
        a, b = list(pool.map(lambda _: run(workers=2), range(2)))
    assert a == b and len(calls(tmp_path)) == 4
    (a.parent / "LO-10.npz").write_bytes(b"truncated")
    assert run() == a
    assert calls(tmp_path).count("LO 10") == 2 and len(calls(tmp_path)) == 5


def test_failed_seed_retry_preserves_completed_runs(example, tmp_path):
    _, run = example
    flag = tmp_path / "fail-13"
    flag.touch()
    with pytest.raises(RuntimeError, match=r"V, seed 13.*nnlojet.log"):
        run()
    assert len(list((tmp_path / "work").rglob("*.npz"))) == 3
    flag.unlink()
    assert run().is_file()
    assert calls(tmp_path) == ["LO 10", "LO 11", "V 12", "V 13", "V 13"]


@pytest.mark.parametrize("changed", ["inputs/grid", "lib/libnnlojet_h.so", "pdfs/TestSet/TestSet_0000.dat",
                                     "pdfs/TestSet/TestSet.info", "bin/NNLOJET", "h.run"])
def test_modified_physics_inputs_start_new_calculation(example, tmp_path, changed):
    _, run = example
    first = run(workers=2)
    path = tmp_path / changed
    if changed == "inputs/grid":
        path.write_text("7")
    else:
        with path.open("a") as stream:
            stream.write("\n" if changed == "h.run" else "\n# changed\n")
    second = run(workers=2)
    assert first.parent != second.parent
    assert len(calls(tmp_path)) == 8
    if changed == "inputs/grid":
        assert json.loads(second.read_text())["integration"]["raw_mean_pb"][0][1] == pytest.approx(.1975)


@pytest.mark.parametrize("defect", ["missing", "ambiguous", "nx", "double", "evaluations", "columns",
                                    "variance", "finite", "incomplete"])
def test_invalid_native_results_never_enter_cache(example, tmp_path, defect):
    _, run = example
    (tmp_path / "defect").write_text(defect)
    with pytest.raises(ValueError, match=r"invalid NNLOJET output.*nnlojet.log"):
        run()
    assert not list((tmp_path / "work").rglob("*.npz"))


@pytest.mark.parametrize(("key", "value"), [
    ("axes", []), ("axes", {}), ("axes", {"y": "two axes"}), ("axes", {"y": "abs_yh", "z": "ptz"}),
    ("scale_labels", "scale"), ("scale_labels", ["two", "two"]), ("scale_labels", ["", "varied"]),
    ("scale_labels", ["only one"]), ("seeds", [1]), ("seeds", [1, True]), ("seeds", [1, 1]),
    ("seeds", [999998, 999999]), ("channels", ["LO", "lo"]), ("channels", [{}]),
    ("dependencies", "inputs/grid"), ("dependencies", ["../grid"]), ("dependencies", ["absent"]),
    ("dependencies", {"moments.run": "inputs/grid"}), ("dependencies", ["inputs/grid", "./inputs/grid"]),
    ("dependencies", {"booking.sha256": "inputs/grid"}), ("runcard", "absent.run")])
def test_invalid_setup_rejected_before_running(example, tmp_path, key, value):
    setup, run = example
    with pytest.raises(ValueError):
        run({**setup, key: value})
    assert not (tmp_path / "calls").exists()


def test_native_regions_are_summed_before_normalization(example, tmp_path):
    setup, run = example
    (tmp_path / "defect").write_text("regions")
    setup = {key: value for key, value in setup.items() if key != "scale_labels"}
    table = json.loads(run(setup).read_text())
    assert table["cross_section_pb"] == pytest.approx(1.71)
    assert table["moments"][0]["value"] == pytest.approx(.1935 / .855)
    assert table["integration"]["scale_labels"] == ["muf=125 mur=125", "muf=250 mur=250"]


def test_explicit_undefined_axis_value_encodes_only_that_axis(example):
    setup, _ = example
    _, plain = _measurements(setup)
    assert plain.splitlines()[-1] == "abs_yh 1 1 0 4 0 0 0 0 0 0"
    for value, token in [(0, "0"), (-.5, "-0.5")]:
        _, explicit = _measurements({**setup, "axes": {"y": {"observable": "ptj2", "undefined": value}}})
        assert explicit.splitlines()[-1] == f"ptj2 1 1 0 4 0 0 0 0 0 0 {token}"


@pytest.mark.parametrize("axis", [{"observable": "ptj2"}, {"observable": "ptj2", "undefined": True},
                                  {"observable": "ptj2", "undefined": float("nan")},
                                  {"observable": "ptj2", "undefined": float("inf")},
                                  {"observable": "ptj2", "undefined": 10**400},
                                  {"observable": "ptj2", "undefined": 0, "other": 1},
                                  {"undefined": 0}])
def test_invalid_undefined_observable_fails_before_integration(example, tmp_path, axis):
    setup, run = example
    with pytest.raises(ValueError):
        run({**setup, "axes": {"y": axis}})
    assert not (tmp_path / "calls").exists()


def test_shared_native_histogram_file_and_mapped_dependency(example, tmp_path):
    setup, run = example
    (tmp_path / "h.run").write_text(CARD.replace("HISTOGRAMS\n", "HISTOGRAMS > combined\n", 1))
    external = tmp_path / "shared"
    (tmp_path / "inputs/grid").rename(external)
    result = run({**setup, "dependencies": {"inputs/grid": str(external)}})
    assert json.loads(result.read_text())["cross_section_pb"] == pytest.approx(.855)


@pytest.mark.skipif(not os.environ.get("MAXENT_TEST_NNLOJET"), reason="requires a patched NNLOJET v1.0.2 and CT18NNLO")
def test_native_born_higgs_recoil_identity(tmp_path):
    # At Born level pT(H)=0 on every phase-space point. Its mapped T1=-1, T2=+1
    # are exact at every scale and for every independent integration seed.
    reference = Path(__file__).resolve().parents[1] / "examples/nnlojet_moments"
    setup = json.loads((reference / "calculation.json").read_text())
    config = deepcopy(setup["config"]["born"]["y_abs"])
    config["range"] = [0, 100]
    setup.update(config={"born": {"pt": config}}, axes={"pt": "pth"}, channels=["LO"],
                 degrees={"pt": [1, 2]}, seeds=[31, 32])
    text = (reference / setup["runcard"]).read_text().replace("20000[5]", "1000[1]").replace("100000[10]", "1000[1]")
    (tmp_path / setup["runcard"]).write_text(text)
    output = calculate(tmp_path / "config.json", setup, binary=os.environ["MAXENT_TEST_NNLOJET"],
                       directory=tmp_path / "work", workers=2)
    table = json.loads(output.read_text())
    rates = np.array(table["integration"]["raw_mean_pb"])
    assert rates.shape == (7, 3) and np.all(rates[:, 0] > 0)
    np.testing.assert_allclose(rates[:, 1], -rates[:, 0], rtol=1e-10)
    np.testing.assert_allclose(rates[:, 2], rates[:, 0], rtol=1e-10)
