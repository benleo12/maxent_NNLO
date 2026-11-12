"""Reweight your prior using a released table.

    python examples/upgrade_with_released_moments.py dy prior.npz [posterior.npz]

The installed command is maxent-upgrade reweight dy prior.npz [posterior.npz].
Use maxent-upgrade inspect dy to see the required arrays and fiducial selection.
The output preserves the input arrays and adds weight_upgraded in pb.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxent_upgrade.cli import main


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2:
        process, prior = args
        args.append(str(Path(prior).with_suffix("")) + f"_{process}_upgraded.npz")
    main(["reweight", *args])
