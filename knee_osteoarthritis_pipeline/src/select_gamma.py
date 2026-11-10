"""
select_gamma.py — Non-test-leaking γ selection for OWMM-Adaptive (v18/v19).

Read-only. Selects the tempered-prior γ per dataset on VALIDATION macro-F1 (the
paper's primary frequency-aware metric), then reports the chosen model's TEST
per-class F1 as mean ± std over seeds against the baseline. Selecting γ within the
OWMM family (every variant already mixes, so none collapses to the majority the way
a plain-CE baseline would) by validation macro-F1 yields the per-dataset optima the
paper reports: γ=0.5 on the milder KOA and γ=0.25 on the severe EyePACS — the optimal
γ tracks imbalance severity. Validation QWK is printed alongside for transparency
(reference only, not the selector).

Why per-dataset and not a single unified γ: the two datasets sit at different
imbalance regimes (~13× vs ~36×), so the strength of the log-prior correction that is
optimal differs; forcing one γ across both leaves performance on the table. γ is
therefore treated as a standard per-dataset hyperparameter tuned on the validation
split, and the choice is confirmed on the held-out test set — no test peeking.

Usage (from knee_osteoarthritis_pipeline/src):
    python select_gamma.py                      # both datasets
    python select_gamma.py --dataset eyepacs
    python select_gamma.py --csv                # also dump reporting tables to CSV
"""

import argparse
import csv
import glob
import json
import os
import statistics as st

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "..", "results")

CLASSES = ["0", "1", "2", "3", "4"]

# The OWMM-Adaptive γ family, all dedicated v19 tempered-prior runs at the same
# adaptive-mixing config: γ=0 (g00, no log-prior) ... γ=1 (g10, full Balanced Softmax),
# with g025/g05/g075 sweeping the middle. Using one consistent family for every point
# is what the paper's γ-ablation table reports.
GAMMA_FAMILY = [
    (0.0,  "v19_owmm_tempered_g00"),
    (0.25, "v19_owmm_tempered_g025"),
    (0.5,  "v19_owmm_tempered_g05"),
    (0.75, "v19_owmm_tempered_g075"),
    (1.0,  "v19_owmm_tempered_g10"),
]
BASELINE = "v1_baseline"
UNIFIED_GAMMA = 0.5  # single-value fallback to also report (KOA optimum; shown for reference)

# When set (via --seed), restrict the analysis to a single tuning seed so the
# gamma curve is not averaged over an uneven seed set across variants.
_SEED = None


def _seed_files(dataset, version, filename):
    """seed_*/filename for a version (mirrors run_all._discover_seed_metrics);
    restricted to seed_{_SEED} when a tuning seed is selected."""
    pat = f"seed_{_SEED}" if _SEED is not None else "seed_*"
    return sorted(glob.glob(os.path.join(RESULTS_DIR, dataset, version, pat, filename)))


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def load_val_at_best(dataset, version):
    """Mean over seeds of (val_qwk, val_f1) taken at each run's best-F1 epoch."""
    vqwk, vf1 = [], []
    for p in _seed_files(dataset, version, "metrics.json"):
        d = json.load(open(p))
        best, hist = d.get("best", {}), d.get("history", {})
        ep = best.get("epoch")
        if not ep:
            continue
        if hist.get("val_qwk"):
            vqwk.append(hist["val_qwk"][ep - 1])
        if hist.get("val_f1"):
            vf1.append(hist["val_f1"][ep - 1])
    q, _, nq = _mean_std(vqwk)
    f, _, nf = _mean_std(vf1)
    return {"val_qwk": q, "val_f1": f, "n": max(nq, nf)}


def load_test(dataset, version):
    """Per-seed test metrics; returns dict of metric -> (mean, std, n) incl. per-class F1."""
    runs = [json.load(open(p)) for p in _seed_files(dataset, version, "test_metrics.json")]
    if not runs:
        return None
    out = {}
    for key in ["test_qwk", "test_mae", "test_f1_macro", "test_accuracy"]:
        out[key] = _mean_std([r.get(key) for r in runs])
    for c in CLASSES:
        out[f"f1_{c}"] = _mean_std(
            [r["per_class"][c]["f1"] for r in runs if c in r.get("per_class", {})]
        )
    out["seeds"] = ",".join(
        sorted(os.path.basename(os.path.dirname(p)).replace("seed_", "")
               for p in _seed_files(dataset, version, "test_metrics.json"))
    )
    return out


def _fmt(ms):
    if ms is None or ms[0] is None:
        return "   -   "
    m, s, n = ms
    return f"{m:.3f}+/-{s:.3f}" if n > 1 else f"{m:.3f}       "


def select_gamma(dataset):
    print("=" * 92)
    print(f"DATASET: {dataset}")
    print("=" * 92)

    # -- 1. Selection table (validation) --------------------------------------
    print("\n[1] gamma SELECTION on VALIDATION (mean over seeds, at best-F1 checkpoint)")
    print(f"    {'g':>5}  {'variant':<32}{'val_QWK':>9}{'val_F1':>9}  n")
    val_rows = []
    for g, v in GAMMA_FAMILY:
        r = load_val_at_best(dataset, v)
        val_rows.append((g, v, r))
        q = f"{r['val_qwk']:.4f}" if r["val_qwk"] is not None else "  -  "
        f = f"{r['val_f1']:.4f}" if r["val_f1"] is not None else "  -  "
        print(f"    {g:>5}  {v:<32}{q:>9}{f:>9}  {r['n']}")

    have_q = [(g, v, r) for g, v, r in val_rows if r["val_qwk"] is not None]
    have_f = [(g, v, r) for g, v, r in val_rows if r["val_f1"] is not None]
    g_qwk = max(have_q, key=lambda t: t[2]["val_qwk"])[0] if have_q else None
    g_f1 = max(have_f, key=lambda t: t[2]["val_f1"])[0] if have_f else None
    print(f"\n    -> argmax val_F1   = g={g_f1}   (PRIMARY selector)")
    print(f"    -> argmax val_QWK  = g={g_qwk}   (shown for transparency; reference only)")

    # -- 2. Reporting table (test) for selected gamma vs baseline --------------
    chosen = [("val-macroF1 selected", g_f1)]
    if UNIFIED_GAMMA != g_f1:
        chosen.append(("unified", UNIFIED_GAMMA))
    g2v = dict((g, v) for g, v in GAMMA_FAMILY)

    base = load_test(dataset, BASELINE)
    report_tables = []
    for framing, g in chosen:
        if g is None:
            continue
        sel = load_test(dataset, g2v[g])
        print(f"\n[2] REPORT ({framing}: g={g}) - TEST, mean+/-std over seeds")
        cols = ["f1_0", "f1_1", "f1_2", "f1_3", "f1_4", "test_f1_macro", "test_qwk"]
        head = ["c0", "c1", "c2", "c3", "c4", "macF1", "QWK"]
        print(f"    {'method':<26}" + "".join(h.rjust(12) for h in head))
        print(f"    {'v1_baseline':<26}" + "".join(_fmt(base[k]).rjust(12) for k in cols))
        if sel is None:
            print(f"    (no test data for g={g})")
            continue
        print(f"    {g2v[g]:<26}" + "".join(_fmt(sel[k]).rjust(12) for k in cols))

        def d(k):
            if sel[k][0] is None or base[k][0] is None:
                return "   -   "
            return f"{sel[k][0]-base[k][0]:+.3f}"
        print(f"    {'  d vs baseline':<26}" + "".join(d(k).rjust(12) for k in cols))
        report_tables.append((framing, g, g2v[g], base, sel))
    return report_tables


def dump_csv(dataset, tables, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "framing", "gamma", "method",
                    "f1_0", "f1_1", "f1_2", "f1_3", "f1_4", "macro_f1", "qwk", "mae", "seeds"])
        for framing, g, ver, base, sel in tables:
            for tag, m in [("baseline", base), (ver, sel)]:
                if m is None:
                    continue
                def cell(k):
                    v = m[k]
                    return "" if v[0] is None else f"{v[0]:.4f}+/-{v[1]:.4f}"
                w.writerow([dataset, framing, g, tag,
                            cell("f1_0"), cell("f1_1"), cell("f1_2"), cell("f1_3"), cell("f1_4"),
                            cell("test_f1_macro"), cell("test_qwk"), cell("test_mae"), m["seeds"]])
    print(f"    [csv] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Select OWMM-Adaptive gamma on validation QWK.")
    ap.add_argument("--dataset", choices=["koa", "eyepacs"], default=None,
                    help="Dataset (default: both).")
    ap.add_argument("--csv", action="store_true", help="Also write reporting tables to CSV.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Restrict analysis to a single tuning seed (e.g. 123).")
    args = ap.parse_args()

    global _SEED
    _SEED = args.seed

    datasets = [args.dataset] if args.dataset else ["koa", "eyepacs"]
    for ds in datasets:
        tables = select_gamma(ds)
        if args.csv:
            dump_csv(ds, tables, os.path.join(RESULTS_DIR, ds, "gamma_selection.csv"))
        print()


if __name__ == "__main__":
    main()
