#!/usr/bin/env python3
"""
Find the single OWMix variant that clears the baseline on BOTH datasets.

Loads test metrics for every version present under results/koa and
results/eyepacs (averaging across all seed_* subdirs), computes each version's
delta vs that dataset's v1_baseline, then cross-references the two datasets to
rank versions by their *worst* (min across datasets) improvement — the variant
with the highest min-delta is the most robust "works on both" candidate.

Usage:
    python analyze_owmix_winner.py
    python analyze_owmix_winner.py --metric test_f1_macro   # rank by F1 (default)
    python analyze_owmix_winner.py --family owmm             # restrict to v17-19
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

DATASETS = ["koa", "eyepacs"]
BASELINE = "v1_baseline"
METRICS = ["test_accuracy", "test_f1_macro", "test_auc_macro"]


def canon(vn):
    """Strip a trailing temperature suffix so '..._t10' matches its base name."""
    return re.sub(r"_t\d\d$", "", vn)


def load_dataset(results_root, ds):
    """Return {canon_version: {metric: mean, 'n_seeds': k, 'raw': dirname}}."""
    root = results_root / ds
    out = {}
    if not root.is_dir():
        return out
    for vdir in sorted(os.listdir(root)):
        vpath = root / vdir
        if not vpath.is_dir() or vdir == "comparison" or vdir.startswith("_"):
            continue
        files = glob.glob(str(vpath / "seed_*" / "test_metrics.json"))
        if not files and (vpath / "test_metrics.json").exists():
            files = [str(vpath / "test_metrics.json")]
        vals = {m: [] for m in METRICS}
        for f in files:
            m = json.load(open(f))
            for k in METRICS:
                if k in m:
                    vals[k].append(m[k])
        if not any(vals[k] for k in METRICS):
            continue
        rec = {"n_seeds": max(len(vals[k]) for k in METRICS), "raw": vdir}
        for k in METRICS:
            rec[k] = sum(vals[k]) / len(vals[k]) if vals[k] else None
        out[canon(vdir)] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(_SCRIPT_DIR / "../results"))
    ap.add_argument("--metric", default="test_f1_macro", choices=METRICS)
    ap.add_argument("--family", default="all",
                    help="'all', or a prefix filter like 'owmm' / 'v1' / 'v18'")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    results_root = Path(args.results_dir)
    data = {ds: load_dataset(results_root, ds) for ds in DATASETS}
    base = {ds: data[ds].get(BASELINE) for ds in DATASETS}

    for ds in DATASETS:
        if base[ds] is None:
            print(f"[WARN] no baseline for {ds}; deltas there will be blank")

    def keep(v):
        if v == BASELINE:
            return False
        if args.family == "all":
            return True
        if args.family == "owmm":
            return v.startswith(("v17_", "v18_", "v19_"))
        return v.startswith(args.family)

    versions = sorted(set(v for ds in DATASETS for v in data[ds] if keep(v)))

    m = args.metric
    print(f"\n# Cross-dataset OWMix winner — ranked by min Δ({m}) across "
          f"{', '.join(d.upper() for d in DATASETS)}\n")
    hdr = ["Version"]
    for ds in DATASETS:
        hdr += [f"{ds} {m.split('_')[-1]}", f"{ds} Δ"]
    hdr += ["min Δ", "both>base?"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))

    rows = []
    for v in versions:
        cells = [v]
        deltas = []
        present_both = True
        for ds in DATASETS:
            rec = data[ds].get(v)
            b = base[ds]
            if rec is None or b is None or rec.get(m) is None:
                cells += ["—", "—"]
                deltas.append(None)
                present_both = False
                continue
            d = rec[m] - b[m]
            deltas.append(d)
            cells += [f"{rec[m]:.4f}", f"{d:+.4f}"]
        valid = [d for d in deltas if d is not None]
        min_d = min(valid) if (valid and present_both) else None
        both = present_both and all(d > 0 for d in valid)
        cells.append(f"{min_d:+.4f}" if min_d is not None else "—")
        cells.append("✅" if both else ("—" if present_both else "n/a"))
        rows.append((min_d if min_d is not None else -1e9, both, cells))

    rows.sort(key=lambda r: (r[1], r[0]), reverse=True)
    for _, _, cells in rows:
        print("| " + " | ".join(cells) + " |")

    winners = [c[2][0] for c in rows if c[1]]
    print()
    if winners:
        best = rows[0][2][0]
        print(f"**Winner (beats baseline on BOTH, best min-Δ {m}): `{best}`**")
        if len(winners) > 1:
            print(f"Other variants clearing both: {', '.join(w for w in winners if w != best)}")
    else:
        print("**No variant beats baseline on BOTH datasets yet** for "
              f"metric {m}. Closest is `{rows[0][2][0]}` (min Δ {rows[0][0]:+.4f}).")
    print("\n_Note: single-seed rows are noisy; confirm the winner with a 2nd seed._")


if __name__ == "__main__":
    main()
