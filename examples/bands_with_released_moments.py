"""Uncertainty re-solves with the same constraints as the central result.

    python examples/bands_with_released_moments.py dy prior.npz [n_boot] [bands.npz]
    python examples/bands_with_released_moments.py wp prior.npz 20 bands.npz --recoil pt_w ctw_window

Saved event weights sum to one. Matching arrays sigma_*_pb provide the total
cross section for absolute predictions, e.g. scale_w * sigma_scale_pb[:, None].
The scale array excludes the central solve; include central in its envelope.
Moment counts stay fixed. A moved window starts from zero multipliers because
its measurement functions can differ. Shower changes require separate prior
weights or samples from the generator and another call to upgrade.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxent_upgrade import upgrade, load_moments, released


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("process", choices=released.processes())
    parser.add_argument("prior")
    parser.add_argument("n_boot", nargs="?", type=int, default=20)
    parser.add_argument("output", nargs="?")
    parser.add_argument("--born", nargs="*")
    parser.add_argument("--recoil", nargs="*")
    parser.add_argument("--mixed", nargs="*")
    parser.add_argument("--nnlo-zj", action="store_true")
    parser.add_argument("--skip-pdf", action="store_true", help="omit the 59 PDF-member solves")
    args = parser.parse_args()
    if args.n_boot < 0:
        parser.error("n_boot must be nonnegative")
    output = args.output or str(Path(args.prior).with_suffix("")) + f"_{args.process}_bands.npz"
    if Path(output).suffix != ".npz":
        parser.error("the output filename must end in .npz")
    if Path(output).exists():
        parser.error("output already exists; choose a different path")
    with np.load(args.prior, allow_pickle=False) as sample:
        events = dict(sample)
    selection = dict(born=args.born, recoil=args.recoil, mixed=args.mixed, nnlo_zj=args.nnlo_zj)
    process = args.process
    moments, config = load_moments(process, **selection)
    central = upgrade(events, moments, config)
    orders = dict(central.report["moment_selection"]["chosen"])
    frozen = {**config, "fixed_orders": orders, "lam0": central.report["lam"]}
    print(f"Central: {central.summary()}")
    saved = dict(central=central.weights, sigma_central_pb=released.cross_section(process))

    scale_weights, scale_rates = [], []
    for scale in range(1, 7):
        shifted, _ = load_moments(process, scale=scale, check_seam=False, **selection)
        result = upgrade(events, shifted, frozen)
        scale_weights.append(result.weights)
        scale_rates.append(released.cross_section(process, scale=scale))
        print(f"Scale {scale}: {result.summary()}")
    saved.update(scale_w=np.asarray(scale_weights), sigma_scale_pb=np.asarray(scale_rates))

    n_runs = released.table(process)["n_independent_runs"]
    rng = np.random.default_rng(1)
    bootstrap_weights, bootstrap_rates = [], []
    for _ in range(args.n_boot):
        runs = rng.integers(0, n_runs, n_runs)
        shifted, _ = load_moments(process, runs=runs, check_seam=False, **selection)
        bootstrap_weights.append(upgrade(events, shifted, frozen).weights)
        bootstrap_rates.append(released.cross_section(process, runs=runs))
    saved.update(boot_w=np.asarray(bootstrap_weights).reshape(args.n_boot, len(central.weights)),
                 sigma_boot_pb=np.asarray(bootstrap_rates))
    print(f"Input statistics: {args.n_boot} resamples of {n_runs} independent runs")
    if args.nnlo_zj:
        print("The NNLO Z+jet input has no per-run replicas; only the Drell-Yan input is resampled.")

    for window in ("down", "up") if released.windows(process) else ():
        if args.nnlo_zj:
            # The letter's NNLO Z+jet tower has no released moved-window moments.
            continue
        shifted, shifted_config = load_moments(process, window=window, **selection)
        result = upgrade(events, shifted, {**shifted_config, "fixed_orders": orders})
        saved[f"window_{window}"] = result.weights
        print(f"Transition {window}: {result.summary()}")
    saved["sigma_window_pb"] = np.asarray(saved["sigma_central_pb"])

    members = released.pdf_table(process)
    if members is not None and not args.skip_pdf:
        pdf_weights, pdf_rates = [central.weights], [released.cross_section(process)]
        for member in range(1, members["n_members"]):
            shifted, shifted_config = load_moments(process, pdf_member=member, **selection)
            result = upgrade(events, shifted, {**shifted_config, "fixed_orders": orders,
                                               "lam0": central.report["lam"]})
            pdf_weights.append(result.weights)
            pdf_rates.append(released.cross_section(process, pdf_member=member))
        saved.update(pdf_w=np.asarray(pdf_weights), sigma_pdf_pb=np.asarray(pdf_rates))
        print(f"PDF: {len(pdf_weights)} members; use released.hessian_band on their histograms")

    np.savez_compressed(output, **saved)
    print(f"Saved {output}: normalized weights and the corresponding cross sections in pb")


if __name__ == "__main__":
    main()
