# BM_OS/src/evaluate.py
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import brier_score_loss
from typing import Optional, List
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

def load_ground_truth(json_path: str) -> dict:
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Build a dict: PatientID -> (duration, event)
    return {item["PatientID"]: (item["OS"]["Duration"], item["OS"]["Event"]) for item in data}

def evaluate_log_risk(pred_csv: str, gt: dict):
    df = pd.read_csv(pred_csv)

    rows = []
    for _, row in df.iterrows():
        pid = str(row["PatientID"])
        if pid not in gt:
            continue
        duration, event = gt[pid]
        score = -float(row["log_risk"])
        rows.append(
            {
                "PatientID": pid,
                "Duration": float(duration),
                "Event": int(event),
                "log_risk": float(row["log_risk"]),
                "score": score,
            }
        )

    detail_df = pd.DataFrame(rows)
    ci = concordance_index(detail_df["Duration"], detail_df["score"], detail_df["Event"])
    print(f"Concordance index (C-index): {ci:.4f}")
    return ci, detail_df


def evaluate_time_auc(pred_csv: str, gt: dict, times: List[float]):
    df = pd.read_csv(pred_csv)
    
    durations = []
    events = []
    risks = []
    
    for _, row in df.iterrows():
        pid = str(row["PatientID"])
        if pid not in gt:
            continue
        duration, event = gt[pid]
        durations.append(duration)
        events.append(bool(event))
        risks.append(-row["log_risk"])  
    
    # Convert to structured array
    y_true = Surv.from_arrays(events, durations)
    risks = np.array(risks)

    print("\n📈 Time-dependent AUCs:")
    for t in times:
        try:
            times_eval = np.array([t])
            _, aucs, _ = cumulative_dynamic_auc(
                y_true, y_true, risks, times_eval
            )
            print(f"AUC at time {t}: {aucs[0]:.4f}")
        except Exception as e:
            print(f"⚠️ Failed to compute AUC at time {t}: {e}")

def evaluate_survival_horizons(pred_csv: str, gt: dict, horizons: List[float]):
    df = pd.read_csv(pred_csv)
    for h in horizons:
        col = f"surv@{int(h)}" if f"surv@{int(h)}" in df.columns else f"surv@{float(h):.2f}"
        if col not in df.columns:
            print(f"Missing survival prediction for horizon {h}")
            continue

        y_true = []
        y_pred = []
        for _, row in df.iterrows():
            pid = str(row["PatientID"])
            if pid not in gt:
                continue
            duration, event = gt[pid]
            y_true.append(1.0 if duration > h else float(1 - event)) 
            y_pred.append(row[col])

        brier = brier_score_loss(y_true, y_pred)
        print(f"Brier score at {h}: {brier:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate survival predictions")
    parser.add_argument("--pred_csv", type=str, required=True, help="Path to prediction CSV")
    parser.add_argument("--json", type=str, required=True, help="Path to ground truth dataset JSON")
    parser.add_argument("--horizons", type=float, nargs='*', default=None, help="Optional horizons for survival evaluation")
    args = parser.parse_args()

    gt = load_ground_truth(args.json)

    basename = os.path.basename(args.json).lower()
    if "train" in basename:
        out = "train"
    elif "val" in basename:
        out = "val"
    elif "test" in basename:
        out = "test"
    else:
        out = os.path.splitext(basename)[0] 
    print("Evaluating log-risk prediction...")

    ci, detail_df = evaluate_log_risk(args.pred_csv, gt)

    if args.horizons:
        print("\n📈 Evaluating survival probabilities...")
        evaluate_survival_horizons(args.pred_csv, gt, args.horizons)

        print("\n📈 Evaluating time-dependent AUCs...")
        evaluate_time_auc(args.pred_csv, gt, args.horizons)

        print("\n📉 Plotting Kaplan-Meier curve...")
    print(f"C-index: {ci}")
    return ci, detail_df

if __name__ == "__main__":
    main()