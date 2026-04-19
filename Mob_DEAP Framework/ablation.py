"""
evaluation/ablation.py  —  Independent ablation study  (C0–C5)
evaluation/stats.py     —  Statistical significance testing

Ablation configurations (Reviewer #1 Comment 11, Reviewer #3 Comment 4):
    C0: Baseline  — no novel components
    C1: +HFCBF only
    C2: +BO only
    C3: +EPO + DA-EVA only
    C4: +HFCBF + BO  (no EPO)
    C5: Full model  (all components)

Each configuration runs with num_seeds random seeds.
Results reported as mean ± std with Wilcoxon Signed-Rank p-values.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Callable, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Ablation configuration flags
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_CONFIGS: Dict[str, Dict] = {
    "C0_Baseline"       : dict(hfcbf=False, bo=False, epo=False, da_eva=False),
    "C1_HFCBF_only"     : dict(hfcbf=True,  bo=False, epo=False, da_eva=False),
    "C2_BO_only"        : dict(hfcbf=False, bo=True,  epo=False, da_eva=False),
    "C3_EPO_DA-EVA_only": dict(hfcbf=False, bo=False, epo=True,  da_eva=True),
    "C4_HFCBF_BO"       : dict(hfcbf=True,  bo=True,  epo=False, da_eva=False),
    "C5_Full_Model"     : dict(hfcbf=True,  bo=True,  epo=True,  da_eva=True),
}

FULL_MODEL_KEY = "C5_Full_Model"


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(train_eval_fn: Callable[[Dict, int], Dict],
                 num_seeds    : int = 5,
                 dataset_name : str = "MOT17"
                 ) -> pd.DataFrame:
    """
    Run the independent ablation study.

    Args:
        train_eval_fn:
            Callable(flags: Dict, seed: int) -> Dict with keys:
                "mota", "idf1", "hota", "num_ids"
            Implement this function in train.py using your actual
            training loop and evaluator.

        num_seeds:
            Number of random seeds (default 5; use 10+ for tighter p-values).

        dataset_name:
            Label for console output.

    Returns:
        pd.DataFrame with columns:
            config, mota_mean, mota_std, idf1_mean, idf1_std,
            hota_mean, hota_std, p_vs_full, significant
    """
    from utils.seed import set_seed

    scores: Dict[str, List[Dict]] = {k: [] for k in ABLATION_CONFIGS}

    for seed in range(num_seeds):
        print(f"\n[Ablation] Seed {seed+1}/{num_seeds} — {dataset_name}")
        for cfg_name, flags in ABLATION_CONFIGS.items():
            set_seed(seed)
            result = train_eval_fn(flags, seed)
            scores[cfg_name].append(result)
            print(f"  {cfg_name:30s}  "
                  f"MOTA={result.get('mota', 0):.2f}  "
                  f"IDF1={result.get('idf1', 0):.2f}  "
                  f"HOTA={result.get('hota', 0):.2f}")

    full_motas = [r["mota"] for r in scores[FULL_MODEL_KEY]]
    rows = []

    for cfg_name, results in scores.items():
        motas = [r["mota"] for r in results]
        idf1s = [r["idf1"] for r in results]
        hotas = [r["hota"] for r in results]

        if cfg_name != FULL_MODEL_KEY and len(full_motas) == len(motas):
            try:
                _, p = stats.wilcoxon(full_motas, motas)
            except Exception:
                p = float("nan")
        else:
            p = float("nan")

        rows.append({
            "config"      : cfg_name,
            "hfcbf"       : ABLATION_CONFIGS[cfg_name]["hfcbf"],
            "bo"          : ABLATION_CONFIGS[cfg_name]["bo"],
            "epo"         : ABLATION_CONFIGS[cfg_name]["epo"],
            "da_eva"      : ABLATION_CONFIGS[cfg_name]["da_eva"],
            "mota_mean"   : round(np.mean(motas), 3),
            "mota_std"    : round(np.std(motas),  3),
            "idf1_mean"   : round(np.mean(idf1s), 3),
            "idf1_std"    : round(np.std(idf1s),  3),
            "hota_mean"   : round(np.mean(hotas), 3),
            "hota_std"    : round(np.std(hotas),  3),
            "p_vs_full"   : round(p, 5) if not np.isnan(p) else "ref",
            "significant" : (p < 0.05)  if not np.isnan(p) else None,
        })

    df = pd.DataFrame(rows)
    print("\n── Ablation results ─────────────────────────────────────────")
    print(df[["config","mota_mean","mota_std","idf1_mean",
              "idf1_std","p_vs_full","significant"]].to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Condition-wise evaluation
# ─────────────────────────────────────────────────────────────────────────────
SEQUENCE_CONDITIONS = {
    "MOT17-02": "day_static",    "MOT17-04": "day_static",
    "MOT17-09": "day_static",    "MOT17-05": "day_moving",
    "MOT17-10": "day_moving",    "MOT17-11": "day_moving",
    "MOT17-13": "night_low_light",
    "MOT20-01": "crowded",       "MOT20-02": "crowded",
    "MOT20-03": "crowded",       "MOT20-05": "crowded",
}


def condition_wise_summary(per_seq_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Aggregate per-sequence MOT results by lighting/crowd condition.

    Args:
        per_seq_results:
            {seq_name: {"mota": float, "idf1": float, "hota": float}}
    Returns:
        DataFrame grouped by condition with mean ± std per metric.
    """
    rows = []
    for seq, metrics in per_seq_results.items():
        base = "-".join(seq.split("-")[:2]) if "-" in seq else seq
        cond = SEQUENCE_CONDITIONS.get(base, "normal")
        rows.append({"sequence": seq, "condition": cond, **metrics})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = (df.groupby("condition")[["mota", "idf1", "hota"]]
               .agg(["mean", "std"])
               .round(2))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance  (Tables 11 and 12)
# ─────────────────────────────────────────────────────────────────────────────
def significance_tests(per_model_scores: Dict[str, List[float]],
                       proposed_key    : str = "C5_Full_Model"
                       ) -> pd.DataFrame:
    """
    Wilcoxon Signed-Rank Test (proposed vs each baseline) +
    Friedman Test (all models jointly).

    Args:
        per_model_scores: {model_name: [score_per_fold_or_seed]}
        proposed_key:     key of the proposed model in the dict

    Returns:
        DataFrame with columns:
            baseline, W_stat, p_value, significant, cohens_d, ci_95_lo, ci_95_hi
    """
    proposed = np.array(per_model_scores[proposed_key])
    rows = []

    for name, sc in per_model_scores.items():
        if name == proposed_key:
            continue
        s = np.array(sc)

        # Wilcoxon Signed-Rank test
        try:
            stat, p = stats.wilcoxon(proposed, s)
        except Exception:
            stat, p = float("nan"), float("nan")

        # Cohen's d
        pool_std = np.sqrt((proposed.std()**2 + s.std()**2) / 2 + 1e-9)
        d = (proposed.mean() - s.mean()) / pool_std

        # 95% confidence interval on pairwise differences
        diff = proposed - s
        ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])

        rows.append({
            "baseline"  : name,
            "W_stat"    : round(float(stat), 3),
            "p_value"   : round(float(p),    5),
            "significant": p < 0.05,
            "cohens_d"  : round(float(d),    3),
            "ci_95_lo"  : round(float(ci_lo),3),
            "ci_95_hi"  : round(float(ci_hi),3),
        })

    # Friedman test across all models
    try:
        chi2, p_f = stats.friedmanchisquare(
            *[np.array(v) for v in per_model_scores.values()])
    except Exception:
        chi2, p_f = float("nan"), float("nan")

    print(f"\nFriedman χ² = {chi2:.3f},  p = {p_f:.5f}")
    return pd.DataFrame(rows)
