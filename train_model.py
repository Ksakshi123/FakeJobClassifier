"""Train the fake-job classifier: benchmark model families, keep the best.

Evaluates candidates with stratified out-of-fold probabilities (PR-AUC is the
selection metric because only ~5% of postings are fraudulent), tunes a
decision threshold for the 'fake' class, reports holdout metrics, and saves a
model bundle consumed by predict.py / app.py / evaluate.py.
"""

import joblib
import numpy as np
import pandas as pd
import sklearn.base
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from features import fit_bundle, transform

DATA_PATH = "Dataset/fake_job_postings.csv"
EXTRA_PATH = "Dataset/current/train_extra.csv"  # current-style augmentation data
MODEL_PATH = "Model/model_bundle.pkl"
RANDOM_STATE = 42
N_SPLITS = 3  # fewer folds for the model shootout; final metrics use the holdout


def make_models():
    """Candidate classifiers. All must expose predict_proba."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0),
        "multinomial_nb": MultinomialNB(alpha=0.1),
        "complement_nb": ComplementNB(alpha=0.1),
        "sgd_log": SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-5,
            class_weight="balanced", max_iter=1500, tol=1e-3,
            random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced_subsample",
            max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE),
        # NOTE: SVD+gradient-boosting was tried and dropped — TruncatedSVD on
        # 14k x 40k sparse features took >15 min/fold for no expected gain.
    }


def candidates():
    return make_models()


def oof_probabilities(X, y, model, n_splits=N_SPLITS):
    """Out-of-fold P(fake) via cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    import sklearn.base
    prob = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = sklearn.base.clone(model)
        m.fit(X[tr], y.iloc[tr] if hasattr(y, "iloc") else y[tr])
        if hasattr(m, "predict_proba"):
            prob[va] = m.predict_proba(X[va])[:, 1]
        else:
            d = m.decision_function(X[va])
            prob[va] = 1 / (1 + np.exp(-d))
    return prob


def tune_threshold(y_true, y_prob):
    """Pick the threshold maximizing F1 for the fake class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-9)
    f1 = f1[:-1]  # drop trailing point with no threshold
    return float(thresholds[np.argmax(f1)])


def main():
    # 1. Load + clean
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip().str.lower()
    data = data.fillna("")

    # Augment with current-style postings (synthetic scam patterns +
    # recently scraped legitimate ads) when available.
    import os
    if os.path.exists(EXTRA_PATH):
        extra = pd.read_csv(EXTRA_PATH)
        extra.columns = extra.columns.str.strip().str.lower()
        extra = extra.drop(columns=[c for c in ["source"] if c in extra.columns])
        for col in data.columns:
            if col not in extra.columns:
                extra[col] = ""
        extra = extra[data.columns]
        data = pd.concat([data, extra], ignore_index=True)
        print(f"Augmented with {len(extra)} current-style rows "
              f"({int(extra['fraudulent'].sum())} scam)")

    # Partial-input augmentation: app users often paste ONLY the description.
    # Without this, the model learns "empty structured fields == scam" and
    # flags every description-only legit posting. Blank all fields except
    # description on a sample of legit rows + all fraud rows.
    keep_cols = {"job_id", "fraudulent", "description"}
    blank_cols = [c for c in data.columns if c not in keep_cols]
    rng = np.random.default_rng(RANDOM_STATE)
    legit_idx = data.index[data["fraudulent"] == 0].to_numpy()
    fraud_idx = data.index[data["fraudulent"] == 1].to_numpy()
    n_blank_legit = min(4000, len(legit_idx))
    blanked_idx = np.concatenate([
        rng.choice(legit_idx, size=n_blank_legit, replace=False),
        fraud_idx,
    ])
    blanked = data.loc[blanked_idx].copy()
    blanked[blank_cols] = ""
    data = pd.concat([data, blanked], ignore_index=True)
    print(f"Added {len(blanked)} partial-input (description-only) rows")

    y = data["fraudulent"].astype(int)

    # 2. Holdout split BEFORE fitting any feature builder (no leakage)
    idx_train, idx_test = train_test_split(
        np.arange(len(data)), test_size=0.20,
        random_state=RANDOM_STATE, stratify=y,
    )
    train_df, test_df = data.iloc[idx_train], data.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    # 3. Fit feature pipeline on train only
    X_train, bundle = fit_bundle(train_df)
    X_test = transform(test_df, bundle)
    print(f"Features: train={X_train.shape}, test={X_test.shape}\n")

    # 4. Model shootout via out-of-fold probabilities
    results = {}
    for name, model in candidates().items():
        print(f"Evaluating {name} ...", flush=True)
        prob = oof_probabilities(X_train, y_train, model)
        pr_auc = average_precision_score(y_train, prob)
        roc = roc_auc_score(y_train, prob)
        results[name] = {"pr_auc": pr_auc, "roc_auc": roc, "prob": prob}
        print(f"  PR-AUC={pr_auc:.4f}  ROC-AUC={roc:.4f}", flush=True)

    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\n>>> Best model: {best_name} "
          f"(PR-AUC {results[best_name]['pr_auc']:.4f})")

    best_prob = results[best_name]["prob"]
    threshold = tune_threshold(y_train.values, best_prob)
    print(f">>> Tuned fake-class threshold: {threshold:.3f}")

    # 5. Train the winner on all training data
    model = sklearn.base.clone(candidates()[best_name])
    model.fit(X_train, y_train)
    prob_test = model.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= threshold).astype(int)

    # 6. Holdout evaluation
    print("\n==============================")
    print(f"HOLDOUT PERFORMANCE — {best_name}")
    print("==============================")
    print(f"ROC-AUC: {roc_auc_score(y_test, prob_test):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, prob_test):.4f}")
    print(f"\nAt threshold {threshold:.3f}:")
    print(classification_report(
        y_test, pred_test, target_names=["Real Job", "Potentially Fake"]))
    print("Confusion matrix [real, fake] rows / [pred real, pred fake] cols:")
    print(confusion_matrix(y_test, pred_test))

    # 7. Save bundle
    bundle["model"] = model
    bundle["model_name"] = best_name
    bundle["threshold"] = threshold
    bundle["threshold_medium"] = threshold * 0.6
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel bundle saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
