"""
Phase 1 — Data Loading and Understanding
==========================================
IBM HR Analytics dataset: 1,470 employee records, 35 features.

Outputs
-------
  outputs/fig1_distributions.png       Numerical feature distributions by attrition
  outputs/fig2_correlation_heatmap.png Correlation matrix across all numeric features
  outputs/fig3_categorical_attrition.png Attrition rate by every categorical variable
  Printed dataset profile and key segment findings

Run
---
  python phase1_data_understanding.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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


def load_and_profile(path):
    df = pd.read_csv(path)
    df["Attrition_Binary"] = (df["Attrition"] == "Yes").astype(int)

    print("Dataset profile")
    print(f"  Records          : {len(df):,}")
    print(f"  Features         : {df.shape[1]}")
    print(f"  Attrition rate   : {df['Attrition_Binary'].mean()*100:.1f}%  ({df['Attrition_Binary'].sum()} employees)")
    stay = (df["Attrition"] == "No").sum()
    left = (df["Attrition"] == "Yes").sum()
    print(f"  Class imbalance  : {stay/left:.1f}:1  (stayed vs left)")
    print(f"  Missing values   : {df.isnull().sum().sum()}")

    const_cols = [c for c in df.columns if df[c].nunique() == 1]
    print(f"  Constant columns : {const_cols}  (zero predictive value, will be dropped)")

    print("\nAttrition rate by key segments")
    for col, grp_label in [
        ("OverTime",      "OverTime"),
        ("JobLevel",      "JobLevel"),
        ("MaritalStatus", "MaritalStatus"),
    ]:
        grp = df.groupby(col)["Attrition_Binary"].mean() * 100
        print(f"  {grp_label}:")
        for val, rate in grp.items():
            print(f"    {str(val):<20} {rate:.1f}%")

    return df


def plot_distributions(df, output_dir):
    stayed = df[df["Attrition"] == "No"]
    left   = df[df["Attrition"] == "Yes"]

    num_features = [
        "Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears",
        "YearsSinceLastPromotion", "DistanceFromHome",
        "YearsInCurrentRole", "YearsWithCurrManager",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, feat in enumerate(num_features):
        ax = axes[i]
        ax.hist(stayed[feat], bins=25, alpha=0.65, color=C_STAY, label="Stayed", density=True)
        ax.hist(left[feat],   bins=25, alpha=0.65, color=C_LEFT, label="Left",   density=True)
        ax.axvline(stayed[feat].mean(), color=C_STAY, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axvline(left[feat].mean(),   color=C_LEFT, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(feat, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel("Density" if i % 4 == 0 else "")

    p1 = mpatches.Patch(color=C_STAY, alpha=0.65, label="Stayed (83.9%)")
    p2 = mpatches.Patch(color=C_LEFT, alpha=0.65, label="Left (16.1%)")
    fig.legend(handles=[p1, p2], loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Numerical feature distributions by attrition status", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout(pad=1.5)
    plt.savefig(f"{output_dir}/fig1_distributions.png", dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"\nSaved: {output_dir}/fig1_distributions.png")


def plot_correlation_heatmap(df, output_dir):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "EmployeeNumber"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask  = np.triu(np.ones_like(corr, dtype=bool))
    cmap  = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.85, vmin=-0.85, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                square=True, linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title("Correlation matrix — all numerical features", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_correlation_heatmap.png", dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig2_correlation_heatmap.png")


def plot_categorical_attrition(df, output_dir):
    cat_features = [
        "OverTime", "MaritalStatus", "BusinessTravel", "Department",
        "JobLevel", "Gender", "StockOptionLevel", "WorkLifeBalance",
        "JobInvolvement", "EnvironmentSatisfaction", "JobSatisfaction",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, feat in enumerate(cat_features):
        ax = axes[i]
        grp = df.groupby(feat)["Attrition_Binary"].agg(["mean", "count"]).reset_index()
        grp.columns = [feat, "rate", "count"]
        grp = grp.sort_values("rate", ascending=True)
        grp["rate_pct"] = grp["rate"] * 100
        colors = [C_LEFT if r > 0.20 else C_STAY if r < 0.12 else "#BA7517" for r in grp["rate"]]
        bars = ax.barh(grp[feat].astype(str), grp["rate_pct"], color=colors, alpha=0.85, height=0.6)
        for bar, (_, row) in zip(bars, grp.iterrows()):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{row['rate_pct']:.1f}%", va="center", fontsize=8, color="#444")
        ax.axvline(16.1, color="#888", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("Attrition %")
        ax.set_xlim(0, max(grp["rate_pct"]) * 1.25 + 2)

    for j in range(len(cat_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Attrition rate by categorical feature   (dotted line = overall 16.1%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(pad=1.8)
    plt.savefig(f"{output_dir}/fig3_categorical_attrition.png", dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig3_categorical_attrition.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_profile(DATA_PATH)
    plot_distributions(df, OUTPUT_DIR)
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_categorical_attrition(df, OUTPUT_DIR)
    print("\nPhase 1 complete. Run phase2_eda_cleaning.py next.")
