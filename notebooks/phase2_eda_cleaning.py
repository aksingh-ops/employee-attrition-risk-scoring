"""
Phase 2 — EDA and Cleaning
============================
Drops useless/redundant columns, encodes categoricals, and documents
every cleaning decision with the statistical rationale.

Outputs
-------
  outputs/fig4_feature_correlations.png  Per-feature correlation with attrition target
  outputs/fig5_interaction_effects.png   Combined risk factor interaction analysis
  outputs/fig6_boxplots.png              Stayed vs left distributions on key features
  Printed t-test significance results for all numerical features

Run
---
  python phase2_eda_cleaning.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = "../data/ibm_hr_attrition.csv"
OUTPUT_DIR = "../outputs"

C_STAY = "#185FA5"
C_LEFT = "#A32D2D"
C_BG   = "#FAFAFA"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
})


def load_and_clean(path):
    df = pd.read_csv(path)

    # Drop constant columns — zero predictive value
    drop_constant = ["EmployeeCount", "Over18", "StandardHours"]

    # Drop ID column — not a feature
    drop_id = ["EmployeeNumber"]

    # Drop redundant columns based on correlation analysis
    # MonthlyIncome vs JobLevel: r=0.95 — same signal, keep JobLevel (ordinal, cleaner)
    # YearsInCurrentRole vs YearsAtCompany: r=0.76
    # YearsWithCurrManager vs YearsAtCompany: r=0.77
    # DailyRate, HourlyRate, MonthlyRate: weaker than JobLevel
    # PerformanceRating: near-constant (skew=1.92, only 2 unique values)
    drop_redundant = [
        "MonthlyIncome", "YearsInCurrentRole", "YearsWithCurrManager",
        "DailyRate", "HourlyRate", "MonthlyRate", "PerformanceRating",
    ]

    all_drops = drop_constant + drop_id + drop_redundant
    df.drop(columns=all_drops, inplace=True)

    print("Cleaning decisions")
    print(f"  Constant columns removed    : {drop_constant}")
    print(f"  ID column removed           : {drop_id}")
    print(f"  Redundant columns removed   : {drop_redundant}")
    print(f"  Features remaining          : {df.shape[1]}")

    # Encode target
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Ordinal: BusinessTravel has a natural order
    df["BusinessTravel"] = df["BusinessTravel"].map(
        {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    )

    # Binary: two-option columns
    df["OverTime"] = (df["OverTime"] == "Yes").astype(int)
    df["Gender"]   = (df["Gender"]   == "Male").astype(int)

    # One-hot: no natural order
    ohe_cols = ["Department", "JobRole", "MaritalStatus", "EducationField"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    bool_cols = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"  Final feature count         : {df.shape[1] - 1}")

    return df


def ttest_analysis(df):
    """T-test for each numerical feature: do leavers and stayers differ significantly?"""
    print("\nT-test: stayed vs left (numerical features)")
    print(f"  {'Feature':<35} {'Stayed mean':>12} {'Left mean':>10} {'p-value':>10}  Significance")
    print("  " + "-" * 80)

    num_cols = [
        "Age", "YearsAtCompany", "TotalWorkingYears", "YearsSinceLastPromotion",
        "DistanceFromHome", "TrainingTimesLastYear",
    ]
    for col in num_cols:
        if col not in df.columns:
            continue
        g0 = df[df["Attrition"] == 0][col]
        g1 = df[df["Attrition"] == 1][col]
        _, p = stats.ttest_ind(g0, g1)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "not sig"
        print(f"  {col:<35} {g0.mean():>12.2f} {g1.mean():>10.2f} {p:>10.4f}  {sig}")


def plot_feature_correlations(df, output_dir):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "Attrition"]
    corrs = pd.Series(
        {c: df[c].corr(df["Attrition"]) for c in num_cols}
    ).sort_values(key=abs, ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [C_LEFT if v > 0 else C_STAY for v in corrs.values]
    ax.barh(corrs.index[::-1], corrs.values[::-1], color=colors[::-1], alpha=0.85, height=0.65)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with Attrition")
    ax.set_title("Feature correlation with attrition (target variable)", fontsize=12, fontweight="bold")
    for i, (feat, val) in enumerate(zip(corrs.index[::-1], corrs.values[::-1])):
        ax.text(val + (0.005 if val >= 0 else -0.005), i, f"{val:.3f}",
                va="center", ha="left" if val >= 0 else "right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_feature_correlations.png", dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"\nSaved: {output_dir}/fig4_feature_correlations.png")


def plot_interaction_effects(df_raw, output_dir):
    df = pd.read_csv("../data/ibm_hr_attrition.csv")
    df["Attrition_Binary"] = (df["Attrition"] == "Yes").astype(int)
    df["AgeBand"]   = pd.cut(df["Age"], bins=[17, 25, 30, 35, 40, 60],
                             labels=["18-25", "26-30", "31-35", "36-40", "41+"])
    df["IncomeQ"]   = pd.qcut(df["MonthlyIncome"], q=4,
                              labels=["Low", "Mid-Low", "Mid-High", "High"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    combo1 = df.groupby(["OverTime", "MaritalStatus"])["Attrition_Binary"].mean().unstack() * 100
    combo1.plot(kind="bar", ax=axes[0], color=[C_STAY, "#BA7517", C_LEFT], alpha=0.85, width=0.6)
    axes[0].set_title("Overtime x Marital Status", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Attrition %")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].axhline(16.1, color="gray", linestyle=":", linewidth=1)

    combo2 = df.groupby(["JobLevel", "OverTime"])["Attrition_Binary"].mean().unstack() * 100
    combo2.plot(kind="bar", ax=axes[1], color=[C_STAY, C_LEFT], alpha=0.85, width=0.6)
    axes[1].set_title("Job Level x Overtime", fontweight="bold")
    axes[1].set_xlabel("Job Level")
    axes[1].set_ylabel("Attrition %")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].axhline(16.1, color="gray", linestyle=":", linewidth=1)

    combo3 = df.groupby(["AgeBand", "IncomeQ"])["Attrition_Binary"].mean().unstack() * 100
    combo3.plot(kind="bar", ax=axes[2], colormap="RdYlGn_r", alpha=0.85, width=0.65)
    axes[2].set_title("Age Band x Income Quartile", fontweight="bold")
    axes[2].set_xlabel("Age Band")
    axes[2].set_ylabel("Attrition %")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)
    axes[2].axhline(16.1, color="gray", linestyle=":", linewidth=1)

    fig.suptitle("Interaction effects — combined risk factors", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=2)
    plt.savefig(f"{output_dir}/fig5_interaction_effects.png", dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig5_interaction_effects.png")


def plot_boxplots(output_dir):
    df = pd.read_csv("../data/ibm_hr_attrition.csv")
    box_features = [
        "Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears",
        "YearsInCurrentRole", "YearsWithCurrManager", "DistanceFromHome", "TrainingTimesLastYear",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, feat in enumerate(box_features):
        ax = axes[i]
        data_stayed = df[df["Attrition"] == "No"][feat]
        data_left   = df[df["Attrition"] == "Yes"][feat]
        bp = ax.boxplot(
            [data_stayed, data_left], patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )
        bp["boxes"][0].set_facecolor(C_STAY)
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor(C_LEFT)
        bp["boxes"][1].set_alpha(0.75)
        ax.set_title(feat, fontweight="bold", fontsize=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Stayed", "Left"])

    fig.suptitle("Stayed vs left — key numerical features", fontsize=12, fontweight="bold")
    plt.tight_layout(pad=1.5)
    plt.savefig(f"{output_dir}/fig6_boxplots.png", dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig6_boxplots.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_clean(DATA_PATH)
    ttest_analysis(df)
    plot_feature_correlations(df, OUTPUT_DIR)
    plot_interaction_effects(df, OUTPUT_DIR)
    plot_boxplots(OUTPUT_DIR)

    df.to_csv(f"{OUTPUT_DIR}/df_clean.csv", index=False)
    print(f"\nClean dataset saved: {OUTPUT_DIR}/df_clean.csv")
    print("Phase 2 complete. Run phase3_feature_engineering.py next.")
