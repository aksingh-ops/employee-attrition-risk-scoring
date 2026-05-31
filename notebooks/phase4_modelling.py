"""
Phase 4 — Modelling, Evaluation, and SHAP Explainability
==========================================================
Trains Logistic Regression (baseline), Random Forest, and XGBoost on the
SMOTE-balanced training set. Evaluates all three on the held-out test set.
Runs SHAP TreeExplainer on the winning model and builds the risk tier output.

Model selection rationale
--------------------------
Recall is the primary metric. A missed leaver costs the firm $50K-$200K in
replacement. A false alarm costs one unnecessary HR conversation. The cost
asymmetry is roughly 10:1 to 40:1, so the model that catches more actual
leavers wins even if it generates some additional false positives.

Results (held-out test set, threshold=0.38)
--------------------------------------------
  Logistic Regression : AUC 0.715  Recall 0.489
  XGBoost             : AUC 0.746  Recall 0.404
  Random Forest       : AUC 0.748  Recall 0.511  <-- winner

Outputs
-------
  outputs/fig8_model_comparison.png   ROC curves, metric bars, confusion matrix
  outputs/fig9_shap_analysis.png      Global SHAP importance + individual waterfall
  outputs/attrition_risk_scores.csv   294 employees scored with tier + SHAP driver

Run
---
  python phase4_modelling.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve,
)
from xgboost import XGBClassifier
import shap

ARTIFACTS  = "../outputs/phase3_artifacts.pkl"
OUTPUT_DIR = "../outputs"
THRESHOLD  = 0.38
RANDOM_SEED = 42

C_LR  = "#534AB7"
C_RF  = "#A32D2D"
C_XGB = "#3B6D11"
C_BG  = "#FAFAFA"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
})


def load_artifacts(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_models(d):
    print("Training models on SMOTE-balanced training set...")

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    lr.fit(d["X_train_sc"], d["y_train_sm"])
    print("  [1/3] Logistic Regression done")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=4,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(d["X_train_sm"], d["y_train_sm"])
    print("  [2/3] Random Forest done")

    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5.2,
        eval_metric="auc", random_state=RANDOM_SEED, verbosity=0,
    )
    xgb.fit(d["X_train_sm"], d["y_train_sm"])
    print("  [3/3] XGBoost done")

    return lr, rf, xgb


def evaluate_all(lr, rf, xgb, d):
    results = []
    configs = [
        (lr,  d["X_test_sc"], "Logistic Regression"),
        (rf,  d["X_test"],    "Random Forest"),
        (xgb, d["X_test"],    "XGBoost"),
    ]
    y_test = d["y_test"]

    for model, X_eval, name in configs:
        proba = model.predict_proba(X_eval)[:, 1]
        pred  = (proba >= THRESHOLD).astype(int)
        results.append({
            "name":      name,
            "auc":       roc_auc_score(y_test, proba),
            "recall":    recall_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "f1":        f1_score(y_test, pred),
            "proba":     proba,
            "cm":        confusion_matrix(y_test, pred),
        })

    print("\nModel comparison (held-out test set)")
    print(f"  {'Model':<24} {'AUC':>7} {'Recall':>8} {'Precision':>10} {'F1':>7}")
    print("  " + "-" * 58)
    for r in results:
        star = " *" if r["name"] == "Random Forest" else ""
        print(f"  {r['name']:<24} {r['auc']:>7.4f} {r['recall']:>8.4f} {r['precision']:>10.4f} {r['f1']:>7.4f}{star}")

    rf_r = next(r for r in results if r["name"] == "Random Forest")
    cm = rf_r["cm"]
    print(f"\nRandom Forest confusion matrix (threshold={THRESHOLD})")
    print(f"  Correctly flagged: {cm[1,1]} of {cm[1,:].sum()} actual leavers  (recall {rf_r['recall']:.3f})")
    print(f"  False alarms:      {cm[0,1]}")

    return results


def run_shap(rf, d, n_samples=150):
    print("\nRunning SHAP TreeExplainer on Random Forest...")
    X_sample   = d["X_test"].iloc[:n_samples]
    explainer  = shap.TreeExplainer(rf)
    sv         = explainer.shap_values(X_sample)

    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]

    importance = pd.Series(
        np.abs(sv).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)

    print("  Top 5 attrition drivers:")
    for feat, val in importance.head(5).items():
        print(f"    {feat:<35} {val:.4f}")

    return sv, importance


def build_risk_output(d, rf_proba, sv, feature_names, output_dir):
    def top_driver(row):
        idx = int(np.argmax(np.abs(row)))
        direction = "raises" if row[idx] > 0 else "lowers"
        return f"{feature_names[idx]} ({direction} risk)"

    n = min(len(sv), len(d["X_test"]))
    drivers = [top_driver(sv[i]) for i in range(n)]
    if len(drivers) < len(d["X_test"]):
        drivers += ["—"] * (len(d["X_test"]) - len(drivers))

    out = d["X_test"].copy()
    out["employee_id"]           = [f"EMP-{str(i).zfill(4)}" for i in range(len(out))]
    out["attrition_probability"] = rf_proba.round(4)
    out["risk_score_pct"]        = (rf_proba * 100).round(1)
    out["risk_tier"]             = pd.cut(
        rf_proba, bins=[0, 0.30, 0.55, 1.0], labels=["Low", "Medium", "High"]
    ).astype(str)
    out["actual_attrition"]      = d["y_test"].values
    out["top_shap_driver"]       = drivers

    print("\nRisk tier validation")
    for tier in ["High", "Medium", "Low"]:
        sub  = out[out["risk_tier"] == tier]
        rate = sub["actual_attrition"].mean() * 100
        print(f"  {tier:<8}: {len(sub):>3} employees  |  actual attrition: {rate:.1f}%")

    out.to_csv(f"{output_dir}/attrition_risk_scores.csv", index=False)
    print(f"\nRisk scores saved: {output_dir}/attrition_risk_scores.csv")
    return out


def plot_model_comparison(results, d, output_dir):
    y_test = d["y_test"]
    colors = {"Logistic Regression": C_LR, "Random Forest": C_RF, "XGBoost": C_XGB}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ROC curves
    ax = axes[0]
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax.plot(fpr, tpr, color=colors[r["name"]], lw=1.8,
                label=f"{r['name']} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=1, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — all 3 models", fontweight="bold")
    ax.legend(fontsize=8)

    # Metric bars
    ax = axes[1]
    metrics = ["auc", "recall", "precision", "f1"]
    mlabels = ["AUC-ROC", "Recall", "Precision", "F1"]
    mcols   = ["#185FA5", "#A32D2D", "#BA7517", "#3B6D11"]
    x = np.arange(len(results))
    w = 0.2
    for i, (metric, lbl, col) in enumerate(zip(metrics, mlabels, mcols)):
        ax.bar(x + i*w - 0.3, [r[metric] for r in results], w,
               color=col, alpha=0.82, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"][:3] for r in results])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.90)
    ax.set_title("Metric comparison", fontweight="bold")
    ax.legend(fontsize=8)

    # Confusion matrix — RF winner
    ax = axes[2]
    rf_r = next(r for r in results if r["name"] == "Random Forest")
    cm = rf_r["cm"]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred: Stay", "Pred: Leave"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual: Stay", "Actual: Leave"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    fontweight="bold", color="white" if cm[i, j] > cm.max()*0.5 else "#333")
    ax.set_title(f"Confusion Matrix — Random Forest\nAUC={rf_r['auc']:.3f}  Recall={rf_r['recall']:.3f}",
                 fontweight="bold", color=C_RF)

    fig.suptitle("Phase 4 — Model Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=2)
    plt.savefig(f"{output_dir}/fig8_model_comparison.png", dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig8_model_comparison.png")


def plot_shap_analysis(importance, sv, d, rf_proba, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Global importance
    ax = axes[0]
    top15 = importance.head(15).sort_values(ascending=True)
    q75   = top15.quantile(0.75)
    q40   = top15.quantile(0.40)
    colors = ["#A32D2D" if v > q75 else "#185FA5" if v > q40 else "#888780"
              for v in top15.values]
    ax.barh(top15.index, top15.values, color=colors, alpha=0.85, height=0.65)
    ax.axvline(top15.values.mean(), color="#888", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 attrition drivers — SHAP (Random Forest)", fontweight="bold")
    p1 = mpatches.Patch(color="#A32D2D", alpha=0.85, label="Top drivers")
    p2 = mpatches.Patch(color="#185FA5", alpha=0.85, label="Medium impact")
    p3 = mpatches.Patch(color="#888780", alpha=0.85, label="Lower impact")
    ax.legend(handles=[p1, p2, p3], fontsize=8, loc="lower right")

    # Waterfall: highest risk employee
    ax2 = axes[1]
    high_risk_idx  = int(np.argmax(rf_proba[:len(sv)]))
    shap_person    = sv[high_risk_idx]
    feat_shap_df   = pd.DataFrame({"feature": list(d["X_test"].columns), "shap": shap_person})
    feat_shap_df   = feat_shap_df.reindex(feat_shap_df["shap"].abs().sort_values(ascending=False).index).head(15)
    feat_shap_df   = feat_shap_df.sort_values("shap")
    colors_wf = ["#A32D2D" if v > 0 else "#185FA5" for v in feat_shap_df["shap"]]
    ax2.barh(feat_shap_df["feature"], feat_shap_df["shap"], color=colors_wf, alpha=0.85, height=0.65)
    ax2.axvline(0, color="#444", linewidth=0.8)
    ax2.set_xlabel("SHAP value  (red = pushes toward leaving  |  blue = pushes toward staying)")
    risk_pct = rf_proba[high_risk_idx]
    ax2.set_title(f"SHAP waterfall — highest-risk employee\nPredicted attrition probability: {risk_pct:.1%}",
                  fontweight="bold", color="#A32D2D")
    for i, (_, row) in enumerate(feat_shap_df.iterrows()):
        offset = 0.001 if row["shap"] >= 0 else -0.001
        ax2.text(row["shap"] + offset, i, f"{row['shap']:+.3f}", va="center",
                 ha="left" if row["shap"] >= 0 else "right", fontsize=8)

    fig.suptitle("SHAP Explainability — why the model flags each employee",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(pad=2)
    plt.savefig(f"{output_dir}/fig9_shap_analysis.png", dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig9_shap_analysis.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    d = load_artifacts(ARTIFACTS)
    lr, rf, xgb = train_models(d)
    results = evaluate_all(lr, rf, xgb, d)

    rf_proba = next(r for r in results if r["name"] == "Random Forest")["proba"]
    sv, importance = run_shap(rf, d)

    build_risk_output(d, rf_proba, sv, d["feature_names"], OUTPUT_DIR)
    plot_model_comparison(results, d, OUTPUT_DIR)
    plot_shap_analysis(importance, sv, d, rf_proba, OUTPUT_DIR)

    print("\nPhase 4 complete. All outputs saved to outputs/")
