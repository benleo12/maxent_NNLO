"""Fit the frozen independent W-study targets on a compatible prior.

    python examples/w_joint_refinement.py wp my_wp_prior.npz wp_selected
    python examples/w_joint_refinement.py wm my_wm_prior.npz wm_joint --model J2

Canonical input arrays are weight, m_w, y_abs, pt_w and ctw1, already in the
training table's fiducial selection. The authors' built-prior files can also be
read; their selection and angle are constructed explicitly below.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxent_upgrade import Variation, upgrade_ensemble, weight_diagnostics
from maxent_upgrade.provenance import canonical_json, file_hash, table_identity
if __package__:
    from .w_joint_inputs import load_study
else:
    from w_joint_inputs import load_study


DATA = Path(__file__).resolve().parent / "data/w_joint"


def lepton_eta(events):
    """Keep signed pseudorapidity; accept the massless-lepton y_l convention."""
    if "eta_l" not in events:
        if "l_born" in events:
            momentum = events["l_born"]
            events["eta_l"] = np.arcsinh(momentum[:, 2] / np.hypot(momentum[:, 0], momentum[:, 1]))
        elif "y_l" in events:
            events["eta_l"] = np.asarray(events["y_l"])
    return events


def prepare_events(source, charge):
    """Accept selected canonical arrays or select the authors' built prior."""
    canonical = {"weight", "m_w", "y_abs", "pt_w", "ctw1"}
    if canonical.issubset(source):
        return lepton_eta(dict(source)), "input supplied in the table's fiducial selection"
    required = {"weight", "m_w", "y_w", "pt_w", "pt_l", "y_l", "pt_nu", "mt", "charge", "l_born", "nu"}
    if not required.issubset(source):
        raise ValueError("provide canonical selected arrays weight, m_w, y_abs, pt_w and ctw1")
    selected = ((source["pt_l"] > 20) & (abs(source["y_l"]) < 2.5) &
                (source["pt_nu"] > 25) & (source["mt"] > 40) &
                (source["charge"] == (1 if charge == "wp" else -1)))
    count = len(selected)
    events = {key: value[selected] if value.ndim and len(value) == count else value
              for key, value in source.items()}
    lepton, neutrino = events["l_born"], events["nu"]
    pair = lepton + neutrino
    mass2 = pair[:, 3]**2 - np.sum(pair[:, :3]**2, axis=1)
    if (mass2 <= 0).any():
        raise ValueError("selected lepton-neutrino pairs need positive invariant mass")
    qt2 = pair[:, 0]**2 + pair[:, 1]**2
    numerator = ((lepton[:, 3] + lepton[:, 2]) * (neutrino[:, 3] - neutrino[:, 2]) -
                 (lepton[:, 3] - lepton[:, 2]) * (neutrino[:, 3] + neutrino[:, 2]))
    costh = numerator / np.sqrt(mass2 * (mass2 + qt2))
    costh = np.clip(np.where(pair[:, 2] < 0, -costh, costh), -1, 1)
    events["ctw1"] = 1 + costh
    events["y_abs"] = abs(events["y_w"])
    events["source_index"] = np.flatnonzero(selected)
    events.setdefault("event_id", events["source_index"])
    return lepton_eta(events), "pt_l>20, pt_nu>25, mt>40 GeV; |y_l|<2.5; requested charge; pre-QED leptons"


def run(charge, prior, output, *, model="selected", data=DATA, bootstrap=20, seed=1):
    data, output, prior = Path(data), Path(output), Path(prior)
    if bootstrap < 2:
        raise ValueError("at least two independent-run replicas are required")
    output.mkdir(parents=True, exist_ok=False)
    table_path = data / f"w_{charge}_training.json"
    runs_path = data / f"w_{charge}_training_runs.npz"
    table = json.loads(table_path.read_text())
    with np.load(prior, allow_pickle=False) as source:
        events, selection = prepare_events(dict(source), charge)
    moments, config, sigma = load_study(charge, model=model, directory=data)
    scales = []
    for scale in range(1, 7):
        varied, _, rate = load_study(charge, model=model, scale=scale, directory=data)
        scales.append(Variation(moments=varied, cross_section=rate, label=f"scale {scale}"))
    with np.load(runs_path, allow_pickle=False) as source:
        accumulators = source["acc"]
    # Each channel is an independent integration; all booked observables within
    # a channel share a seed and must be resampled together.
    if not np.allclose(accumulators.mean(axis=1).sum(axis=0), table["accumulator_mean_pb"], rtol=1e-10, atol=1e-9):
        raise ValueError("training runs do not reproduce the table's full mean accumulator vector")
    rng = np.random.default_rng(seed)

    def replicas():
        for index in range(bootstrap):
            raw = sum(channel[rng.integers(0, len(channel), len(channel))].mean(axis=0)[:, 0]
                      for channel in accumulators)
            varied, _, rate = load_study(charge, model=model, raw=raw, directory=data)
            yield Variation(moments=varied, cross_section=rate, label=f"integration replica {index + 1}")

    result = upgrade_ensemble(events, moments, config, {"scale": scales, "stat": replicas()},
                              cross_section=sigma,
                              progress=lambda label, fit: print(f"{label}: {fit.summary()}", flush=True))
    result.notes.extend([
        "This frozen study supplies scale and target-integration variations, with no PDF, shower or moved-window members.",
        "Histogram sample errors condition on the fitted weights; they do not include a resampled-prior refit.",
    ])
    result.provenance["moment_table"] = table_identity(table_path.name, table, {"model": model})
    result.provenance["input_files"] = [dict(name=p.name, sha256=file_hash(p)) for p in (prior, table_path, runs_path)]
    result.provenance["study"] = dict(model=table["selected"] if model == "selected" else model,
                                      event_selection=selection, bootstrap_count=bootstrap, bootstrap_seed=seed,
                                      propagation="independent channel-wise resampling of complete seed accumulators")
    np.savez_compressed(output / "posterior.npz", **events, **result.arrays())
    for observable, bins in (("eta_l", np.linspace(-2.5, 2.5, 21)), ("pt_l", np.linspace(20, 100, 33))):
        if observable in events:
            np.savez_compressed(output / (observable + ".npz"), **result.histogram(events[observable], bins))
    (output / "statistics.json").write_text(canonical_json(weight_diagnostics(result.central.weights, events["weight"])) + "\n")
    print(f"Saved {output}. The independent validation data were not opened during this fit.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("charge", choices=["wp", "wm"])
    parser.add_argument("prior")
    parser.add_argument("output")
    parser.add_argument("--model", choices=["selected", "B", "A", "J2", "J4"], default="selected")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    run(args.charge, args.prior, args.output, model=args.model, data=args.data,
        bootstrap=args.bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
