"""Evaluate the trained model on an independently labeled CSV of job postings.

Usage:
    python evaluate.py path/to/labeled_jobs.csv

The CSV must contain the same columns as Dataset/fake_job_postings.csv
(including a 'fraudulent' column with 0/1 labels). This is the honest way to
check whether the model generalizes to current, real-world listings rather
than just the Kaggle dataset it was trained on.
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from features import transform

BUNDLE_PATH = "Model/model_bundle.pkl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Path to labeled job postings CSV")
    args = parser.parse_args()

    bundle = joblib.load(BUNDLE_PATH)
    model, threshold = bundle["model"], bundle["threshold"]

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in ["fraudulent"] if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing required column(s): {missing}")
    y = df["fraudulent"].astype(int)

    X = transform(df, bundle)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    print(f"Evaluated {len(df)} postings "
          f"({int(y.sum())} labeled fraudulent, {y.mean() * 100:.1f}%)")
    print(f"Model threshold: {threshold:.3f}\n")

    try:
        print(f"ROC-AUC: {roc_auc_score(y, prob):.4f}")
        print(f"PR-AUC:  {average_precision_score(y, prob):.4f}")
    except ValueError as e:
        print(f"AUC unavailable (need both classes present): {e}")

    print("\nClassification report:")
    print(classification_report(y, pred, target_names=["Real Job", "Potentially Fake"]))
    print("Confusion matrix [real, fake] x [pred real, pred fake]:")
    print(confusion_matrix(y, pred))

    # Show the most confident mistakes for inspection
    df = df.assign(prob=prob, pred=pred)
    errors = df[df["prob"].round(3).between(0.35, 0.85)]
    if len(errors):
        print(f"\n{len(errors)} borderline cases (probability 0.35-0.85) "
              "— review these manually if possible:")
        for _, row in errors.head(10).iterrows():
            label = "FAKE" if row["fraudulent"] else "real"
            print(f"  prob={row['prob']:.2f} true={label} title={str(row.get('title', ''))[:60]!r}")


if __name__ == "__main__":
    main()
