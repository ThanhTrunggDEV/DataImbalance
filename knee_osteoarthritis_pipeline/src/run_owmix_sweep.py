"""
Run all OWMix temperature variants in sequence.
Usage:
    cd src
    python run_owmix_sweep.py                         # default 30 epochs, GPU 0
    python run_owmix_sweep.py --cuda 1                # override GPU
    python run_owmix_sweep.py --epochs 50 --cuda 1    # custom epochs
"""

import argparse
import os, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from configs.config import VERSIONS
from run_experiment import run_version


def parse_args():
    parser = argparse.ArgumentParser(description="Run all OWMix temperature sweep variants")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cuda", type=int, default=0)
    return parser.parse_args()


def make_args(epochs=30, cuda=0):
    """Build a namespace-compatible args object for run_version()."""
    return argparse.Namespace(
        epochs=epochs,
        cuda=cuda,
        batch_size=None,
        data_dir=None,
        results_dir=None,
        mixup_temperature=None,
        supcon_epochs=None,
        probe_epochs=None,
        finetune_epochs=None,
    )


def main():
    cli_args = parse_args()
    args = make_args(epochs=cli_args.epochs, cuda=cli_args.cuda)

    sweep_versions = [
        v for v in VERSIONS
        if v["name"].startswith("v11_owmixup") or v["name"].startswith("v12_owmixup")
    ]
    sweep_versions.sort(key=lambda v: v["name"])

    print(f"OWMix temperature sweep: {len(sweep_versions)} variants")
    print(f"Epochs: {args.epochs}, CUDA: {args.cuda}")
    print()

    for vcfg in sweep_versions:
        print(f"\n{'='*70}")
        print(f"  RUNNING: {vcfg['name']}  ({vcfg['display']})")
        print(f"{'='*70}")
        try:
            run_version(vcfg, args)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
