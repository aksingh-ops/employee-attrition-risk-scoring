"""
Phase 3 — Feature Engineering and Class Balancing
===================================================
Splits data 80/20, applies SMOTE on training set only, and scales features.
The test set is locked here and never touched again until Phase 4 evaluation.

Why SMOTE on training data only
--------------------------------
If SMOTE is applied before the split, synthetic minority samples generated
from test-set records leak into training. The model would effectively have
seen the test set during training, inflating every evaluation metric. The
test set must reflect real-world class distribution (16.1% attrition) to
produce valid scores.

Outputs
-------
  outputs/fig7_smote_engineering.png   Before/after SMOTE + PCA view of synthetic points
  outputs/phase3_artifacts.pkl         Saved X_train, X_test, y_train, y_test, scaler

Run
---
  python phase3_feature_engineering.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = "../data/ibm_hr_attrition.csv"
OUTPUT_DIR = "../outputs"
RANDOM_SEED = 42
TEST_SIZE   = 0.20

C_STAY = "#185FA5"
C_LEFT = "#A32D2D"
C_BG   = "#FAFAFA"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
})


def load_and_engineer(path):
    df = pd.read_csv(path)

    # Drop useless and redundant columns
    drop_cols = [
        "EmployeeCount", "Over18", "StandardHours", "EmployeeNumber",
        "MonthlyIncome", "YearsInCurrentRole", "YearsWithCurrManager",
        "DailyRate", "HourlyRate", "MonthlyRate", "PerformanceRating",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Encode target
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Ordinal
    df["BusinessTravel"] = df["BusinessTravel"].map(
        {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    )

    # Binary
    df["OverTime"] = (df["OverTime"] == "Yes").astype(int)
    df["Gender"]   = (df["Gender"]   == "Male").astype(int)

    # One-hot
    ohe_cols = ["Department", "JobRole", "MaritalStatus", "EducationField"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    df[df.select_dtypes(bool).columns] = df.select_dtypes(bool).astype(int)

    return df


def split_and_balance(df):
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    print("Train / test split")
    print(f"  Training rows   : {len(X_train):,}")
    print(f"  Test rows       : {len(X_test):,}  (locked until Phase 4)")
    print(f"  Train attrition : {y_train.mean()*100:.1f}%")
    print(f"  Test attrition  : {y_test.mean()*100:.1f}%")

    imbalance = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"\nClass imbalance before SMOTE: {imbalance:.1f}:1")

    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    after = pd.Series(y_train_sm).value_counts()
    print(f"After SMOTE: {after[0]} stayed / {after[1]} left  (1:1)")
    print(f"Training rows after SMOTE: {len(X_train_sm):,}")

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train_sm), columns=X_train_sm.columns
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    return X_train, X_train_sm, X_train_sc, X_test, X_test_sc, y_train, y_train_sm, y_test, scaler


def plot_smote_engineering(X_train, y_train, X_train_sm, y_train_sm, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Bar: before vs after
    ax = axes[0]
    before_counts = [
        (y_train == 0).sum(), (y_train == 1).sum()
    ]
    after_counts = [
        (pd.Series(y_train_sm) == 0).sum(),
        (pd.Series(y_train_sm) == 1).sum()
    ]
    x = np.arange(2)
    w = 0.35
    b1 = ax.bar(x - w/2, before_counts, w, label="Before SMOTE", color=C_STAY, alpha=0.75)
    b2 = ax.bar(x + w/2, after_counts,  w, label="After SMOTE",  color=C_LEFT, alpha=0.75)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{int(bar.get_height())}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Stayed (0)", "Left (1)"])
    ax.set_ylabel("Count")
    ax.set_title("SMOTE: class balance", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(after_counts) * 1.18)

    # Feature count pipeline
    ax = axes[1]
    stages  = ["Raw\nfeatures", "After\ndropping", "After\nOHE", "Final\n(scaled)"]
    counts  = [35, 24, 37, 36]
    colors_ = ["#888780", C_LEFT, C_STAY, "#3B6D11"]
    bars2   = ax.bar(stages, counts, color=colors_, alpha=0.82, width=0.55)
    for bar, val in zip(bars2, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Feature count")
    ax.set_title("Feature count through pipeline", fontweight="bold")
    ax.set_ylim(0, 45)

    # PCA view of synthetic vs real
    ax = axes[2]
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    all_pca = pca.fit_transform(X_train_sm)
    n_real  = len(X_train)

    ax.scatter(all_pca[:n_real][pd.Series(y_train_sm[:n_real]).values == 0, 0],
               all_pca[:n_real][pd.Series(y_train_sm[:n_real]).values == 0, 1],
               alpha=0.22, color=C_STAY, s=12, label="Stayed (real)")
    ax.scatter(all_pca[:n_real][pd.Series(y_train_sm[:n_real]).values == 1, 0],
               all_pca[:n_real][pd.Series(y_train_sm[:n_real]).values == 1, 1],
               alpha=0.50, color=C_LEFT, s=16, label="Left (real)")
    ax.scatter(all_pca[n_real:, 0], all_pca[n_real:, 1],
               alpha=0.40, color="#F0997B", s=14, marker="^", label="Left (SMOTE synthetic)")
    ax.set_title("PCA: real vs SMOTE synthetic", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, markerscale=1.3)

    fig.suptitle("Phase 3 — Feature engineering and SMOTE", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=2)
    plt.savefig(f"{output_dir}/fig7_smote_engineering.png", dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig7_smote_engineering.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_engineer(DATA_PATH)

    X_tr, X_tr_sm, X_tr_sc, X_te, X_te_sc, y_tr, y_tr_sm, y_te, scaler = split_and_balance(df)

    plot_smote_engineering(X_tr, y_tr, X_tr_sm, y_tr_sm, OUTPUT_DIR)

    with open(f"{OUTPUT_DIR}/phase3_artifacts.pkl", "wb") as f:
        pickle.dump({
            "X_train": X_tr, "X_train_sm": X_tr_sm, "X_train_sc": X_tr_sc,
            "X_test": X_te,   "X_test_sc": X_te_sc,
            "y_train": y_tr,  "y_train_sm": y_tr_sm, "y_test": y_te,
            "scaler": scaler, "feature_names": list(X_te.columns),
        }, f)
    print(f"Artifacts saved: {OUTPUT_DIR}/phase3_artifacts.pkl")
    print("Phase 3 complete. Run phase4_modelling.py next.")
