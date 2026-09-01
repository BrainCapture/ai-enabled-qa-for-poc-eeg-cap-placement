#!/usr/bin/env python3
"""
EEG cap placement study: summary statistics and distribution plots.
Reads the published CSV dataset in clinical/data/.
Outputs plots to clinical/outputs/.

Column naming convention
------------------------
*_dev      : signed deviation  = measured − expected  (directional bias)
*_true_abs : unsigned error    = |measured − expected| (accuracy magnitude)

The source column "Absolut afvigelse" is stored as signed (m − e), not |m − e|.
Distribution plots (01, 02, 08) use signed deviations to show directional bias.
Accuracy / comparison plots (04, 05, 07, 10–14) use true_abs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
PLOT_DIR = Path(__file__).parent / "outputs"
PLOT_DIR.mkdir(exist_ok=True)

ELEC_KEYS = ["T7", "T8", "Fp1_lat", "Fp2_lat", "Fp1_ap", "Fp2_ap",
             "O1_lat", "O2_lat", "O1_ap", "O2_ap"]

ELECTRODE_LABELS: dict[str, str] = {
    "T7": "T7", "T8": "T8",
    "Fp1_lat": "Fp1 (lat)", "Fp2_lat": "Fp2 (lat)",
    "Fp1_ap":  "Fp1 (A-P)", "Fp2_ap":  "Fp2 (A-P)",
    "O1_lat":  "O1 (lat)",  "O2_lat":  "O2 (lat)",
    "O1_ap":   "O1 (A-P)",  "O2_ap":   "O2 (A-P)",
}

ELEC_ORDER  = ELEC_KEYS
LABEL_ORDER = [ELECTRODE_LABELS[e] for e in ELEC_ORDER]

CONCLUSION_PALETTE = {"Korrekt": "#4CAF50", "Suboptimalt": "#FF9800", "Forkert": "#F44336"}
CONCLUSION_ORDER   = ["Korrekt", "Suboptimalt", "Forkert"]
CONCLUSION_EN      = {"Korrekt": "Optimal", "Suboptimalt": "Usable", "Forkert": "Incorrect"}

METHOD_DISPLAY = {
    "Pro": "Expert", "Self + App": "Self-guided", "Other + App": "Helper-guided"
}

STYLE_RC = {
    "font.family":       "STIXGeneral",
    "font.size":         13,
    "axes.titleweight":  "normal",
    "axes.labelsize":    13,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   11,
}

METHOD_SIMPLE_ORDER = ["Pro", "App"]
METHOD_THREE_ORDER  = ["Pro", "Self + App", "Other + App"]
METHOD_SIMPLE_PAL   = {"Pro": "#0077BB", "App": "#CC3311"}
METHOD_THREE_PAL    = {"Pro": "#0077BB", "Self + App": "#EE7733", "Other + App": "#009988"}

METHOD_LABEL = {
    "Expert Placement":                     "Pro",
    "Self-placement + Visual Guidance":     "Self + App",
    "Inexperienced User + Visual Guidance": "Other + App",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _read_csv(name: str) -> pd.DataFrame:
    """Read a dataset CSV with exact float64 parsing.

    pandas' default CSV float parser is fast but not correctly rounded, and
    loses the last bit on values like ``0.19999999999999973``. That is enough
    to flip a tie in the Wilcoxon sensitivity analysis and shift its statistic
    by 0.5, so the published p-value would not reproduce.
    """
    return pd.read_csv(DATA_DIR / name, float_precision="round_trip")


def _parse_metadata() -> pd.DataFrame:
    df = _read_csv("participants.csv")
    df = df.dropna(subset=["subject_id"]).copy()
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
    df = df.dropna(subset=["subject_id"])
    df["subject_id"] = df["subject_id"].astype(int)

    # "-" is a sentinel for "not recorded"
    for col in ["hair_texture", "hair_density", "hair_diameter", "hair_length", "hair_styling"]:
        df[col] = df[col].replace("-", np.nan)

    def simplify(col: pd.Series, kw: str, pos: str, neg: str) -> pd.Series:
        return col.apply(
            lambda x: pos if isinstance(x, str) and kw in x
            else (neg if isinstance(x, str) else np.nan)
        )

    df["hair_texture_s"]  = simplify(df["hair_texture"],  "Straight", "Straight/Wavy", "Curly/Coily")
    df["hair_density_s"]  = simplify(df["hair_density"],  "Thin",     "Thin/Average",  "High Density")
    df["hair_diameter_s"] = simplify(df["hair_diameter"], "Fine",     "Fine/Medium",   "Coarse")
    df["hair_length_s"]   = simplify(df["hair_length"],   "Short",    "Short/Shaved",  "Medium/Long")
    df["hair_styling_s"]  = simplify(df["hair_styling"],  "Loose",    "Loose",         "Fixed (braids / locs)")
    return df


def _parse_trial_order() -> pd.DataFrame:
    """Which placement method each subject received in each trial."""
    return _read_csv("trial_order.csv")


def _parse_results() -> pd.DataFrame:
    """Per-trial signed and percentage electrode deviations, plus the
    blinded rater's overall conclusion."""
    return _read_csv("measurements.csv")


def _add_true_abs(df: pd.DataFrame) -> pd.DataFrame:
    """Add *_true_abs columns = |signed deviation|."""
    for e in ELEC_KEYS:
        df[f"{e}_true_abs"] = df[f"{e}_dev"].abs()
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    print("=" * 60)
    print("  EEG CAP PLACEMENT STUDY — SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nParticipants : {df['subject_id'].nunique()}")
    print(f"Total trials : {len(df)}")
    print(f"Age          : mean {meta['age'].mean():.1f}  median {meta['age'].median():.1f}"
          f"  range [{meta['age'].min():.0f}–{meta['age'].max():.0f}]")
    print(f"Sex          : {meta['sex'].value_counts().to_dict()}")

    print("\n--- Conclusion distribution ---")
    counts = df["conclusion"].value_counts().reindex(CONCLUSION_ORDER, fill_value=0)
    for c, n in counts.items():
        print(f"  {c:<30s}: {n:2d} / {len(df)}  ({n/len(df)*100:.0f}%)")

    print("\n--- Conclusion by trial ---")
    for trial in (1, 2):
        sub = df[df["trial"] == trial]
        c_counts = sub["conclusion"].value_counts().reindex(CONCLUSION_ORDER, fill_value=0)
        print(f"  Trial {trial}: " + "  ".join(f"{c}={n}" for c, n in c_counts.items()))

    print("\n--- Per-electrode signed deviation (cm)  [measured − expected] ---")
    print(f"  {'Electrode':<15} {'mean':>7} {'median':>8} {'std':>6} {'min':>7} {'max':>6}")
    for e in ELEC_ORDER:
        v = df[f"{e}_dev"].dropna()
        print(f"  {ELECTRODE_LABELS[e]:<15} {v.mean():>+7.3f} {v.median():>+8.3f} "
              f"{v.std():>6.3f} {v.min():>+7.3f} {v.max():>+6.3f}")

    print("\n--- Per-electrode |error| (cm)  [true absolute] ---")
    print(f"  {'Electrode':<15} {'mean':>6} {'median':>8} {'std':>6} {'max':>6}")
    for e in ELEC_ORDER:
        v = df[f"{e}_true_abs"].dropna()
        print(f"  {ELECTRODE_LABELS[e]:<15} {v.mean():>6.3f} {v.median():>8.3f} "
              f"{v.std():>6.3f} {v.max():>6.3f}")
    print()


# ---------------------------------------------------------------------------
# Long-format helpers
# ---------------------------------------------------------------------------

def _long_dev(df: pd.DataFrame) -> pd.DataFrame:
    """Signed deviations — for bias / distribution plots."""
    return (df[[f"{e}_dev" for e in ELEC_ORDER]]
            .rename(columns={f"{e}_dev": ELECTRODE_LABELS[e] for e in ELEC_ORDER})
            .melt(var_name="Electrode", value_name="Deviation (cm)"))


def _long_pct(df: pd.DataFrame) -> pd.DataFrame:
    pct = (df[[f"{e}_pct_dev" for e in ELEC_ORDER]]
           .rename(columns={f"{e}_pct_dev": ELECTRODE_LABELS[e] for e in ELEC_ORDER})) * 100
    return pct.melt(var_name="Electrode", value_name="Deviation (%)")


def _long_true_abs(df: pd.DataFrame) -> pd.DataFrame:
    """True absolute errors — for accuracy / comparison plots."""
    return (df[[f"{e}_true_abs" for e in ELEC_ORDER]]
            .rename(columns={f"{e}_true_abs": ELECTRODE_LABELS[e] for e in ELEC_ORDER})
            .melt(var_name="Electrode", value_name="|Error| (cm)"))


def _build_long_by_method(df: pd.DataFrame, method_col: str) -> pd.DataFrame:
    frames = []
    for e in ELEC_ORDER:
        tmp = df[[method_col, f"{e}_true_abs", f"{e}_pct_dev"]].copy()
        tmp.columns = [method_col, "|Error| (cm)", "Deviation (%)"]
        tmp["Deviation (%)"] *= 100
        tmp["Electrode"] = ELECTRODE_LABELS[e]
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _conclusion_bars(ax: plt.Axes, df: pd.DataFrame, group_col: str,
                     order: list[str], title: str) -> None:
    ct = (df.groupby(group_col)["conclusion"]
          .value_counts().unstack(fill_value=0)
          .reindex(index=order, columns=CONCLUSION_ORDER, fill_value=0))
    n_by_group = df.groupby(group_col).size().reindex(order, fill_value=0)
    bottoms = np.zeros(len(order))
    x = np.arange(len(order))
    for outcome in CONCLUSION_ORDER:
        vals = ct[outcome].values if outcome in ct.columns else np.zeros(len(order))
        bars = ax.bar(x, vals, bottom=bottoms,
                      color=CONCLUSION_PALETTE[outcome], label=outcome,
                      edgecolor="white", width=0.5)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v > 0:
                pct = v / n_by_group.iloc[i] * 100
                ax.text(bar.get_x() + bar.get_width() / 2, bottoms[i] + v / 2,
                        f"{v}\n({pct:.0f}%)", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={n_by_group[g]})" for g in order], fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=9)


# ---------------------------------------------------------------------------
# Overview plots 01–08
# ---------------------------------------------------------------------------

def plot_signed_deviation_boxplot(df: pd.DataFrame) -> None:
    """Plot 01 — signed deviations, shows directional bias."""
    with plt.rc_context(STYLE_RC):
        fig, ax = plt.subplots(figsize=(12, 5))
        long = _long_dev(df)
        sns.boxplot(data=long, x="Electrode", y="Deviation (cm)",
                    order=LABEL_ORDER, ax=ax, linewidth=1.2,
                    flierprops=dict(marker="o", markersize=4, alpha=0.6))
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("")
        ax.set_ylabel("Signed deviation (cm)")
        plt.xticks(rotation=20, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "01_signed_deviation_boxplot.png", dpi=150)
        plt.close()
    print("  Saved 01_signed_deviation_boxplot.png")


def plot_pct_deviation_violin(df: pd.DataFrame) -> None:
    """Plot 02 — signed percentage deviations."""
    fig, ax = plt.subplots(figsize=(12, 5))
    long = _long_pct(df)
    sns.violinplot(data=long, x="Electrode", y="Deviation (%)",
                   order=LABEL_ORDER, ax=ax, inner="box", cut=0, linewidth=1.1)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_title(f"Signed Percentage Deviation per Electrode — All Trials (n={len(df)})", fontsize=12)
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_pct_deviation_violin.png", dpi=150)
    plt.close()
    print("  Saved 02_pct_deviation_violin.png")


def plot_conclusion_distribution(df: pd.DataFrame) -> None:
    """Plot 03."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n = len(df)
    overall = df["conclusion"].value_counts().reindex(CONCLUSION_ORDER, fill_value=0)
    bars = axes[0].bar(CONCLUSION_ORDER, overall.values,
                       color=[CONCLUSION_PALETTE[c] for c in CONCLUSION_ORDER],
                       edgecolor="white", width=0.55)
    for bar, v in zip(bars, overall.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                     f"{v}  ({v/n*100:.0f}%)", ha="center", va="bottom", fontsize=10)
    axes[0].set_title(f"Overall Outcome (n={n} trials)", fontsize=12)
    axes[0].set_ylim(0, overall.max() + 5)
    axes[0].set_ylabel("Count")

    trial_counts = (df.groupby("trial")["conclusion"].value_counts()
                    .unstack(fill_value=0)
                    .reindex(columns=CONCLUSION_ORDER, fill_value=0))
    trial_counts.plot(kind="bar", ax=axes[1],
                      color=[CONCLUSION_PALETTE[c] for c in CONCLUSION_ORDER],
                      rot=0, edgecolor="white", width=0.55)
    axes[1].set_title("Outcome by Trial", fontsize=12)
    axes[1].set_xlabel("Trial")
    axes[1].legend(title="", loc="upper right", fontsize=9)
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_conclusion_distribution.png", dpi=150)
    plt.close()
    print("  Saved 03_conclusion_distribution.png")


def plot_heatmap(df: pd.DataFrame) -> None:
    """Plot 04 — mean |error| per electrode × participant."""
    heatmap_data = pd.DataFrame(
        {ELECTRODE_LABELS[e]: df.groupby("subject_id")[f"{e}_true_abs"].mean()
         for e in ELEC_ORDER}
    ).T
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(heatmap_data, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.4, cbar_kws={"label": "Mean |error| (cm)", "shrink": 0.8})
    ax.set_title("Mean |Error| per Electrode per Participant", fontsize=12)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_heatmap_per_participant.png", dpi=150)
    plt.close()
    print("  Saved 04_heatmap_per_participant.png")


def plot_sex_comparison(df: pd.DataFrame) -> None:
    """Plot 05 — mean |error| by sex."""
    mean_by_sex = pd.DataFrame(
        {ELECTRODE_LABELS[e]: df.groupby("sex")[f"{e}_true_abs"].mean() for e in ELEC_ORDER}
    ).T
    x, w = np.arange(len(mean_by_sex)), 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, mean_by_sex.get("Male",   pd.Series([0]*len(mean_by_sex))),
           w, label="Male",   color="#5C85D6", edgecolor="white")
    ax.bar(x + w/2, mean_by_sex.get("Female", pd.Series([0]*len(mean_by_sex))),
           w, label="Female", color="#E8738A", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(mean_by_sex.index, rotation=20, ha="right")
    ax.set_ylabel("Mean |error| (cm)")
    ax.set_title("Mean |Error| by Sex", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_sex_comparison.png", dpi=150)
    plt.close()
    print("  Saved 05_sex_comparison.png")


def plot_hair_vs_outcome(df: pd.DataFrame) -> None:
    """Plot 06."""
    hair_vars = [("hair_texture_s", "Hair Texture"), ("hair_density_s", "Hair Density"),
                 ("hair_diameter_s", "Strand Diameter"), ("hair_length_s", "Hair Length")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.flatten(), hair_vars):
        ct = (df.groupby(col)["conclusion"].value_counts().unstack(fill_value=0)
              .reindex(columns=CONCLUSION_ORDER, fill_value=0))
        ct.plot(kind="bar", ax=ax,
                color=[CONCLUSION_PALETTE[c] for c in CONCLUSION_ORDER],
                rot=0, edgecolor="white", width=0.55)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.legend(title="", fontsize=8)
    plt.suptitle("Outcome by Hair Characteristics", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_hair_vs_outcome.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 06_hair_vs_outcome.png")


def plot_age_analysis(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Plot 07 — age distribution + age vs mean |error|."""
    mean_err_per_subj = (
        df.assign(mean_err=lambda d: d[[f"{e}_true_abs" for e in ELEC_ORDER]].mean(axis=1))
        .groupby("subject_id")["mean_err"].mean()
    )
    age_series = meta.set_index("subject_id")["age"].reindex(mean_err_per_subj.index)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(meta["age"], bins=12, color="#7B9BD6", edgecolor="white")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Participant Age Distribution", fontsize=12)

    axes[1].scatter(age_series, mean_err_per_subj,
                    color="#7B9BD6", edgecolors="#3A5A9B", s=70, alpha=0.85, zorder=3)
    valid = age_series.notna() & mean_err_per_subj.notna()
    if valid.sum() > 2:
        m, b = np.polyfit(age_series[valid], mean_err_per_subj[valid], 1)
        xline = np.linspace(age_series.min(), age_series.max(), 100)
        r, p = stats.pearsonr(age_series[valid], mean_err_per_subj[valid])
        axes[1].plot(xline, m * xline + b, color="#E05252", lw=1.5, ls="--",
                     label=f"r = {r:.2f}, p = {p:.3f}")
        axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Mean |error| (cm)")
    axes[1].set_title("Age vs. Mean |Error|", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_age_analysis.png", dpi=150)
    plt.close()
    print("  Saved 07_age_analysis.png")


def plot_trial_comparison(df: pd.DataFrame) -> None:
    """Plot 08 — signed deviation by trial (bias view)."""
    frames = []
    for e in ELEC_ORDER:
        tmp = df[["trial", f"{e}_dev"]].copy()
        tmp.columns = ["trial", "Deviation (cm)"]
        tmp["Electrode"] = ELECTRODE_LABELS[e]
        tmp["Trial"] = tmp["trial"].map({1: "Trial 1", 2: "Trial 2"})
        frames.append(tmp)
    long = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.boxplot(data=long, x="Electrode", y="Deviation (cm)",
                hue="Trial", order=LABEL_ORDER, ax=ax, linewidth=1.1,
                flierprops=dict(marker="o", markersize=3, alpha=0.5),
                palette={"Trial 1": "#5C85D6", "Trial 2": "#E8738A"})
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_title("Signed Deviation per Electrode: Trial 1 vs. Trial 2", fontsize=12)
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    ax.legend(title="", loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "08_trial_comparison.png", dpi=150)
    plt.close()
    print("  Saved 08_trial_comparison.png")


# ---------------------------------------------------------------------------
# Cohen's Kappa and McNemar
# ---------------------------------------------------------------------------

def _cohen_kappa(y1: list, y2: list,
                 categories: list, weights: str = "none") -> float:
    """Weighted or unweighted Cohen's Kappa for two paired ordinal sequences."""
    k   = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    n   = len(y1)
    conf = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        conf[idx[a], idx[b]] += 1.0

    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if weights == "linear":
                w[i, j] = abs(i - j) / (k - 1)
            elif weights == "quadratic":
                w[i, j] = (abs(i - j) / (k - 1)) ** 2
            else:
                w[i, j] = 0.0 if i == j else 1.0

    row_m    = conf.sum(axis=1)
    col_m    = conf.sum(axis=0)
    expected = np.outer(row_m, col_m) / n

    obs = (w * conf).sum() / n
    exp = (w * expected).sum() / n
    return float("nan") if exp == 0 else 1.0 - obs / exp


def _mcnemar(y1: list, y2: list, positive: str) -> tuple[float, float]:
    """McNemar test on paired binary outcomes. Returns (χ², p-value)."""
    b = sum(1 for a, b_ in zip(y1, y2) if a == positive and b_ != positive)
    c = sum(1 for a, b_ in zip(y1, y2) if a != positive and b_ == positive)
    denom = b + c
    if denom == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / denom  # continuity-corrected
    p    = stats.chi2.sf(chi2, df=1)
    return chi2, p


def _stuart_maxwell(y1: list, y2: list, categories: list) -> tuple[float, int, float]:
    """Stuart-Maxwell test for marginal homogeneity in a k×k paired table.

    Drops any category with zero marginal in both methods before solving,
    which handles sparse subgroups where a category never appears.
    Returns (Q, df, p-value).
    """
    k_full = len(categories)
    idx    = {c: i for i, c in enumerate(categories)}
    conf   = np.zeros((k_full, k_full), dtype=float)
    for a, b in zip(y1, y2):
        conf[idx[a], idx[b]] += 1.0

    row_m = conf.sum(axis=1)
    col_m = conf.sum(axis=0)

    # Keep only categories present in at least one margin
    keep = [i for i in range(k_full) if row_m[i] > 0 or col_m[i] > 0]
    if len(keep) < 2:
        return float("nan"), 0, float("nan")

    conf  = conf[np.ix_(keep, keep)]
    row_m = conf.sum(axis=1)
    col_m = conf.sum(axis=0)
    k     = len(keep)

    d = (row_m - col_m)[: k - 1]
    S = np.zeros((k - 1, k - 1))
    for i in range(k - 1):
        S[i, i] = row_m[i] + col_m[i] - 2 * conf[i, i]
        for j in range(i + 1, k - 1):
            S[i, j] = S[j, i] = -(conf[i, j] + conf[j, i])

    try:
        Q = float(d @ np.linalg.solve(S, d))
    except np.linalg.LinAlgError:
        return float("nan"), k - 1, float("nan")
    p = stats.chi2.sf(Q, df=k - 1)
    return Q, k - 1, p


def print_kappa_analysis(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("  COHEN'S KAPPA — METHOD AGREEMENT ON PLACEMENT OUTCOME")
    print("=" * 60)

    # Pivot so each row = one subject, columns = Pro / App outcome
    pivot = (df[df["method_simple"].isin(["Pro", "App"])]
               .pivot(index="subject_id", columns="method_simple",
                      values="conclusion")
               .dropna())
    n_pairs = len(pivot)
    pro_vec = pivot["Pro"].tolist()
    app_vec = pivot["App"].tolist()

    kappa_lw       = _cohen_kappa(pro_vec, app_vec, CONCLUSION_ORDER, weights="linear")
    Q, df_sm, p_sm = _stuart_maxwell(pro_vec, app_vec, CONCLUSION_ORDER)

    # Build confusion counts
    conf_counts: dict[tuple[str, str], int] = {}
    for r in CONCLUSION_ORDER:
        for c in CONCLUSION_ORDER:
            conf_counts[(r, c)] = sum(
                1 for a, b in zip(pro_vec, app_vec) if a == r and b == c)

    print(f"\nPro vs App  (n = {n_pairs} paired participants)")
    print(f"  Stuart-Maxwell test (all 3 categories, df={df_sm}):")
    print(f"    Q={Q:.3f}, p={p_sm:.4f}")
    print(f"  Cohen's κ (linear weights): {kappa_lw:+.3f}")

    print("\n  Confusion table (rows=Pro, cols=App):")
    header = f"  {'':14}" + "".join(f"  {c:<14}" for c in CONCLUSION_ORDER)
    print(header)
    for r in CONCLUSION_ORDER:
        row_str = f"  {r:<14}"
        for c in CONCLUSION_ORDER:
            row_str += f"  {conf_counts[(r,c)]:<14}"
        print(row_str)

    print("\n  Marginal totals:")
    row_m = {r: sum(conf_counts[(r, c)] for c in CONCLUSION_ORDER) for r in CONCLUSION_ORDER}
    col_m = {c: sum(conf_counts[(r, c)] for r in CONCLUSION_ORDER) for c in CONCLUSION_ORDER}
    for cat in CONCLUSION_ORDER:
        print(f"    {cat}: Pro={row_m[cat]}  App={col_m[cat]}  diff={row_m[cat]-col_m[cat]:+d}")

    print("\nPro vs Self+App  /  Pro vs Other+App")
    for app_label in ["Self + App", "Other + App"]:
        piv2 = (df[df["method"].isin(["Pro", app_label])]
                  .pivot(index="subject_id", columns="method", values="conclusion")
                  .dropna())
        if len(piv2) < 3:
            print(f"  {app_label}: too few paired observations")
            continue
        p_vec = piv2["Pro"].tolist()
        a_vec = piv2[app_label].tolist()
        Q_i, df_i, p_i = _stuart_maxwell(p_vec, a_vec, CONCLUSION_ORDER)
        k_lw_i = _cohen_kappa(p_vec, a_vec, CONCLUSION_ORDER, weights="linear")
        print(f"  {app_label} (n={len(piv2)}): Stuart-Maxwell Q={Q_i:.3f} p={p_i:.4f}, "
              f"κ_linear={k_lw_i:+.3f}")
    print()


# ---------------------------------------------------------------------------
# Analysis 1: Pro vs App  (plots 09–11)
# ---------------------------------------------------------------------------

def print_method_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("  ANALYSIS 1 — PRO vs APP")
    print("=" * 60)
    for method in METHOD_SIMPLE_ORDER:
        sub = df[df["method_simple"] == method]
        print(f"\n{method} (n={len(sub)}):")
        counts = sub["conclusion"].value_counts().reindex(CONCLUSION_ORDER, fill_value=0)
        for c, n in counts.items():
            print(f"  {c:<30s}: {n:2d}  ({n/len(sub)*100:.0f}%)")
        print(f"  Mean |error| (all electrodes): "
              f"{np.nanmean(sub[[f'{e}_true_abs' for e in ELEC_ORDER]].values):.3f} cm")

    # Paired Wilcoxon on mean |error|
    true_abs_cols = [f"{e}_true_abs" for e in ELEC_ORDER]
    df2 = df.copy()
    df2["mean_err"] = df2[true_abs_cols].mean(axis=1)
    pro = df2[df2["method_simple"] == "Pro"].set_index("subject_id")["mean_err"]
    app = df2[df2["method_simple"] == "App"].set_index("subject_id")["mean_err"]
    paired = pd.concat([pro.rename("pro"), app.rename("app")], axis=1).dropna()
    stat, p = stats.wilcoxon(paired["pro"], paired["app"])
    print(f"\nWilcoxon signed-rank test (mean |error|, n={len(paired)} pairs):")
    print(f"  W={stat:.1f}, p={p:.4f}")

    # Sensitivity analysis: exclude T7 and T8 (systematic cap-geometry artefact)
    elec_no_t78 = [e for e in ELEC_ORDER if e not in ("T7", "T8")]
    sa_cols = [f"{e}_true_abs" for e in elec_no_t78]
    df3 = df.copy()
    df3["mean_err_s"] = df3[sa_cols].mean(axis=1)
    pro_s = df3[df3["method_simple"] == "Pro"].set_index("subject_id")["mean_err_s"]
    app_s = df3[df3["method_simple"] == "App"].set_index("subject_id")["mean_err_s"]
    paired_s = pro_s.to_frame("pro").join(app_s.to_frame("app"), how="inner").dropna()
    stat_s, p_s = stats.wilcoxon(paired_s["pro"], paired_s["app"])
    print(f"Sensitivity (T7+T8 excluded, n={len(paired_s)} pairs):")
    print(f"  Expert={paired_s['pro'].mean():.3f} cm  App={paired_s['app'].mean():.3f} cm")
    print(f"  W={stat_s:.1f}, p={p_s:.4f}")

    print("\n--- Per-electrode mean |error| (cm) ---")
    print(f"  {'Electrode':<15} {'Pro':>8} {'App':>8} {'Diff':>8}")
    for e in ELEC_ORDER:
        pm = df[df["method_simple"] == "Pro"][f"{e}_true_abs"].mean()
        am = df[df["method_simple"] == "App"][f"{e}_true_abs"].mean()
        print(f"  {ELECTRODE_LABELS[e]:<15} {pm:>8.3f} {am:>8.3f} {am-pm:>+8.3f}")
    print()


def plot_pro_vs_app_conclusion(df: pd.DataFrame) -> None:
    """Plot 09 — donut charts, Expert vs App-guided."""
    from matplotlib.patches import Patch

    RING_INNER  = 0.55
    RING_CENTER = (1.0 + RING_INNER) / 2
    colors      = [CONCLUSION_PALETTE[k] for k in CONCLUSION_ORDER]

    with plt.rc_context(STYLE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))

        for ax, (method, label) in zip(axes, [("Pro", "Expert"), ("App", "App-guided")]):
            sub    = df[df["method_simple"] == method]
            counts = sub["conclusion"].value_counts().reindex(CONCLUSION_ORDER, fill_value=0)
            fracs  = counts.values / counts.sum()

            ax.pie(fracs, colors=colors,
                   wedgeprops=dict(width=1.0 - RING_INNER, edgecolor="white", linewidth=1.5),
                   startangle=90)

            cumulative = 0.0
            for frac in fracs:
                if frac > 0.03:
                    mid = np.deg2rad(90 - (cumulative + frac / 2) * 360)
                    ax.text(RING_CENTER * np.cos(mid), RING_CENTER * np.sin(mid),
                            f"{frac*100:.0f}%",
                            ha="center", va="center",
                            fontsize=13, fontweight="bold", color="white")
                cumulative += frac

            ax.set_aspect("equal")
            ax.set_title(label, pad=8)

        handles = [Patch(facecolor=c, label=CONCLUSION_EN[k])
                   for c, k in zip(colors, CONCLUSION_ORDER)]
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.10, 1, 1])
        plt.savefig(PLOT_DIR / "09_pro_vs_app_conclusion.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close()
    print("  Saved 09_pro_vs_app_conclusion.png")


def plot_pro_vs_app_electrode(df: pd.DataFrame) -> None:
    """Plot 10 — |error| and % deviation by method."""
    long = _build_long_by_method(df, "method_simple")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, ycol, title in [
        (axes[0], "|Error| (cm)",  "|Error| — Pro vs. App"),
        (axes[1], "Deviation (%)", "Signed % Deviation — Pro vs. App"),
    ]:
        sns.boxplot(data=long, x="Electrode", y=ycol,
                    hue="method_simple", hue_order=METHOD_SIMPLE_ORDER,
                    order=LABEL_ORDER, ax=ax, linewidth=1.1,
                    palette=METHOD_SIMPLE_PAL,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("")
        ax.legend(title="", loc="upper right")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "10_pro_vs_app_electrodes.png", dpi=150)
    plt.close()
    print("  Saved 10_pro_vs_app_electrodes.png")


def plot_pro_vs_app_paired(df: pd.DataFrame) -> None:
    """Plot 11 — paired participant-level mean |error|."""
    df2 = df.copy()
    df2["mean_err"] = df2[[f"{e}_true_abs" for e in ELEC_ORDER]].mean(axis=1)
    pro = df2[df2["method_simple"] == "Pro"].set_index("subject_id")["mean_err"]
    app = df2[df2["method_simple"] == "App"].set_index("subject_id")["mean_err"]
    paired = pd.concat([pro.rename("Pro"), app.rename("App")], axis=1).dropna()

    stat, p = stats.wilcoxon(paired["Pro"], paired["App"])
    p_label = f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"

    # Sensitivity: T7/T8 excluded
    elec_no_t78 = [e for e in ELEC_ORDER if e not in ("T7", "T8")]
    df3 = df.copy()
    df3["mean_err_s"] = df3[[f"{e}_true_abs" for e in elec_no_t78]].mean(axis=1)
    pro_s = df3[df3["method_simple"] == "Pro"].set_index("subject_id")["mean_err_s"]
    app_s = df3[df3["method_simple"] == "App"].set_index("subject_id")["mean_err_s"]
    paired_s = pro_s.to_frame("Pro").join(app_s.to_frame("App"), how="inner").dropna()
    stat_s, p_s = stats.wilcoxon(paired_s["Pro"], paired_s["App"])
    p_label_s = f"p = {p_s:.3f}" if p_s >= 0.0001 else "p < 0.0001"

    fig, ax = plt.subplots(figsize=(5, 6))
    for _, row in paired.iterrows():
        ax.plot([0, 1], [row["Pro"], row["App"]], color="gray", alpha=0.4, lw=1.2, zorder=1)
    for col, xp in {"Pro": 0, "App": 1}.items():
        vals = paired[col]
        ax.scatter([xp] * len(vals), vals, color=METHOD_SIMPLE_PAL[col],
                   s=55, zorder=3, alpha=0.85, edgecolors="white", linewidth=0.6)
        ax.errorbar(xp, vals.mean(), yerr=vals.sem() * 1.96,
                    fmt="D", color=METHOD_SIMPLE_PAL[col], markersize=9,
                    capsize=6, capthick=2, lw=2, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Pro\n(n={len(paired)})", f"App\n(n={len(paired)})"], fontsize=11)
    ax.set_ylabel("Mean |error| across all electrodes (cm)")
    ax.set_title("Pro vs. App: Per-participant Mean |Error|", fontsize=11)
    ax.set_xlim(-0.4, 1.4)
    ymax = paired.values.max()
    ax.annotate("", xy=(1, ymax + 0.08), xytext=(0, ymax + 0.08),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
    ax.text(0.5, ymax + 0.12,
            f"All 10 elecs: W={stat:.0f}, {p_label}\n"
            f"Excl. T7/T8:  W={stat_s:.0f}, {p_label_s}",
            ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "11_pro_vs_app_paired.png", dpi=150)
    plt.close()
    print("  Saved 11_pro_vs_app_paired.png")


def plot_pro_vs_app_confusion(df: pd.DataFrame) -> None:
    """Plot 15 — confusion matrix: Pro vs App, journal-quality figure."""
    from matplotlib.patches import Rectangle, Patch

    pivot = (df[df["method_simple"].isin(["Pro", "App"])]
               .pivot(index="subject_id", columns="method_simple", values="conclusion")
               .dropna())
    n_pairs = len(pivot)
    pro_vec = pivot["Pro"].tolist()
    app_vec = pivot["App"].tolist()

    k   = len(CONCLUSION_ORDER)
    idx = {c: i for i, c in enumerate(CONCLUSION_ORDER)}
    conf = np.zeros((k, k), dtype=int)
    for a, b in zip(pro_vec, app_vec):
        conf[idx[a], idx[b]] += 1
    row_m = conf.sum(axis=1)
    col_m = conf.sum(axis=0)

    # ColorBrewer-derived, print-safe and colorblind-considerate
    AGREE_HI  = "#2166AC"   # deep blue — dominant agreement cell
    AGREE_MID = "#74B2D8"   # mid blue — smaller agreement cell
    PRO_HI    = "#FDBF6F"   # amber — Pro better, n ≥ 5
    PRO_LO    = "#FEE4C4"   # light amber — Pro better, n < 5
    PRO_INK   = "#7A3500"
    APP_FILL  = "#B2E0D6"   # teal — App better
    APP_INK   = "#1A5E50"
    ZERO_FILL = "#F7F7F7"   # near-white empty cells
    CELL_EDGE = "#BBBBBB"
    WHITE     = "#FFFFFF"
    GHOST_INK = "#AAAAAA"

    max_diag = max((conf[i, i] for i in range(k)), default=1) or 1

    def cell_style(i: int, j: int) -> tuple:
        n = conf[i, j]
        if i == j:
            return (ZERO_FILL, GHOST_INK) if n == 0 else \
                   ((AGREE_HI, WHITE) if n >= 0.7 * max_diag else (AGREE_MID, WHITE))
        if i < j:  # Pro outcome better than App
            return (ZERO_FILL, GHOST_INK) if n == 0 else \
                   ((PRO_HI, PRO_INK) if n >= 5 else (PRO_LO, PRO_INK))
        return (ZERO_FILL, GHOST_INK) if n == 0 else (APP_FILL, APP_INK)

    with plt.rc_context({"axes.facecolor": "white", "figure.facecolor": "white",
                         "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(5.2, 5.0))

        for i in range(k):
            for j in range(k):
                fc, tc = cell_style(i, j)
                n = conf[i, j]
                # Cells share edges — no gap, uniform thin border
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor=fc, edgecolor=CELL_EDGE, linewidth=0.5, zorder=2))
                ax.text(j, i - 0.07, str(n),
                        ha="center", va="center", fontsize=26,
                        color=tc, fontweight="bold" if n > 0 else "normal", zorder=4)
                if n > 0:
                    ax.text(j, i + 0.26, f"{n / n_pairs * 100:.0f}%",
                            ha="center", va="center", fontsize=9,
                            color=tc, alpha=0.85, zorder=4)

        # Clean outer border
        ax.add_patch(Rectangle((-0.5, -0.5), k, k,
                               facecolor="none", edgecolor="#777777",
                               linewidth=0.8, zorder=5))

        cats_en = ["Optimal", "Usable", "Incorrect"]
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(cats_en, fontsize=9.5)
        ax.set_yticklabels(cats_en, fontsize=9.5)

        # Column labels at top — standard convention for method-comparison matrices
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("App-Guided Placement", fontsize=10, labelpad=10)
        ax.set_ylabel("Expert (Pro) Placement", fontsize=10, labelpad=10)

        ax.set_xlim(-0.5, k - 0.5 + 0.58)
        ax.set_ylim(k - 0.5 + 0.50, -0.5 - 0.08)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, labelsize=9.5, colors="#333333")

        # Marginal totals
        for i in range(k):
            ax.text(k - 0.5 + 0.10, i, f"n = {row_m[i]}",
                    ha="left", va="center", fontsize=7.5, color="#888888")
        for j in range(k):
            ax.text(j, k - 0.5 + 0.17, f"n = {col_m[j]}",
                    ha="center", va="top", fontsize=7.5, color="#888888")

        # Horizontal legend below matrix
        ax.legend(handles=[
            Patch(facecolor=AGREE_HI,  edgecolor=CELL_EDGE, linewidth=0.5, label="Agreement"),
            Patch(facecolor=PRO_HI,    edgecolor=CELL_EDGE, linewidth=0.5, label="Pro > App"),
            Patch(facecolor=APP_FILL,  edgecolor=CELL_EDGE, linewidth=0.5, label="App > Pro"),
            Patch(facecolor=ZERO_FILL, edgecolor=CELL_EDGE, linewidth=0.5, label="No observations"),
        ], loc="upper center", bbox_to_anchor=(0.44, -0.04),
           ncol=4, fontsize=8, frameon=True, framealpha=1.0,
           edgecolor="#CCCCCC", handlelength=0.9, handleheight=0.85,
           borderpad=0.5, labelspacing=0.3, columnspacing=0.8)

        plt.tight_layout()
        plt.savefig(PLOT_DIR / "15_pro_vs_app_confusion_matrix.png", dpi=300,
                    facecolor="white", bbox_inches="tight")
        plt.close()
        print("  Saved 15_pro_vs_app_confusion_matrix.png")


def plot_conclusion_panel(df: pd.DataFrame) -> None:
    """Plot 16 — panel: (a) confusion matrix, (b) three-way stacked bar."""
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.gridspec import GridSpec

    # ── shared confusion matrix data ──────────────────────────────────────────
    pivot = (df[df["method_simple"].isin(["Pro", "App"])]
               .pivot(index="subject_id", columns="method_simple", values="conclusion")
               .dropna())
    n_pairs = len(pivot)
    pro_vec  = pivot["Pro"].tolist()
    app_vec  = pivot["App"].tolist()

    k   = len(CONCLUSION_ORDER)
    idx = {c: i for i, c in enumerate(CONCLUSION_ORDER)}
    conf = np.zeros((k, k), dtype=int)
    for a, b in zip(pro_vec, app_vec):
        conf[idx[a], idx[b]] += 1
    row_m = conf.sum(axis=1)
    col_m = conf.sum(axis=0)

    AGREE_HI  = "#2166AC"
    AGREE_MID = "#74B2D8"
    PRO_HI    = "#FDBF6F"
    PRO_LO    = "#FEE4C4"
    PRO_INK   = "#7A3500"
    APP_FILL  = "#B2E0D6"
    APP_INK   = "#1A5E50"
    ZERO_FILL = "#F7F7F7"
    CELL_EDGE = "#BBBBBB"
    WHITE     = "#FFFFFF"
    GHOST_INK = "#AAAAAA"
    max_diag  = max((conf[i, i] for i in range(k)), default=1) or 1

    def cell_style(i: int, j: int) -> tuple:
        n = conf[i, j]
        if i == j:
            return (ZERO_FILL, GHOST_INK) if n == 0 else \
                   ((AGREE_HI, WHITE) if n >= 0.7 * max_diag else (AGREE_MID, WHITE))
        if i < j:
            return (ZERO_FILL, GHOST_INK) if n == 0 else \
                   ((PRO_HI, PRO_INK) if n >= 5 else (PRO_LO, PRO_INK))
        return (ZERO_FILL, GHOST_INK) if n == 0 else (APP_FILL, APP_INK)

    cats_en = ["Optimal", "Usable", "Incorrect"]

    with plt.rc_context(STYLE_RC):
        fig = plt.figure(figsize=(11.5, 5.2), facecolor="white")
        gs  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3], wspace=0.38)

        # ── (a) confusion matrix ──────────────────────────────────────────────
        ax_cm = fig.add_subplot(gs[0])
        ax_cm.set_facecolor("white")

        for i in range(k):
            for j in range(k):
                fc, tc = cell_style(i, j)
                n = conf[i, j]
                ax_cm.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor=fc, edgecolor=CELL_EDGE, linewidth=0.5, zorder=2))
                ax_cm.text(j, i - 0.08, str(n), ha="center", va="center",
                           fontsize=24, color=tc,
                           fontweight="bold" if n > 0 else "normal", zorder=4)
                if n > 0:
                    ax_cm.text(j, i + 0.28, f"{n/n_pairs*100:.0f}%",
                               ha="center", va="center", fontsize=10,
                               color=tc, alpha=0.85, zorder=4)

        ax_cm.add_patch(Rectangle((-0.5, -0.5), k, k,
                                   facecolor="none", edgecolor="#777777",
                                   linewidth=0.8, zorder=5))
        ax_cm.set_xticks(range(k))
        ax_cm.set_yticks(range(k))
        ax_cm.set_xticklabels(cats_en)
        ax_cm.set_yticklabels(cats_en)
        ax_cm.xaxis.set_ticks_position("top")
        ax_cm.xaxis.set_label_position("top")
        ax_cm.set_xlabel("App-Guided", labelpad=10)
        ax_cm.set_ylabel("Expert", labelpad=10)
        ax_cm.set_xlim(-0.5, k - 0.5 + 0.55)
        ax_cm.set_ylim(k - 0.5 + 0.45, -0.5 - 0.08)
        for spine in ax_cm.spines.values():
            spine.set_visible(False)
        ax_cm.tick_params(length=0, colors="#333333")

        for i in range(k):
            ax_cm.text(k - 0.5 + 0.08, i, f"n={row_m[i]}",
                       ha="left", va="center", fontsize=9, color="#888888")
        for j in range(k):
            ax_cm.text(j, k - 0.5 + 0.15, f"n={col_m[j]}",
                       ha="center", va="top", fontsize=9, color="#888888")

        ax_cm.legend(handles=[
            Patch(facecolor=AGREE_HI,  edgecolor=CELL_EDGE, linewidth=0.5, label="Agreement"),
            Patch(facecolor=PRO_HI,    edgecolor=CELL_EDGE, linewidth=0.5, label="Expert > App"),
            Patch(facecolor=APP_FILL,  edgecolor=CELL_EDGE, linewidth=0.5, label="App > Expert"),
            Patch(facecolor=ZERO_FILL, edgecolor=CELL_EDGE, linewidth=0.5, label="None"),
        ], loc="upper center", bbox_to_anchor=(0.44, -0.06),
           ncol=2, fontsize=10, frameon=True, framealpha=1.0,
           edgecolor="#CCCCCC", handlelength=0.9, handleheight=0.85,
           borderpad=0.5, labelspacing=0.3, columnspacing=0.8)

        ax_cm.text(-0.08, 1.04, "(a)", transform=ax_cm.transAxes,
                   fontsize=14, fontweight="bold", va="bottom")

        # ── (b) three-way stacked bar ─────────────────────────────────────────
        ax_bar = fig.add_subplot(gs[1])

        order_display = ["Expert", "Self-guided", "Helper-guided"]
        df_bar = df.copy()
        df_bar["method_display"] = df_bar["method"].map(METHOD_DISPLAY)

        ct = (df_bar.groupby("method_display")["conclusion"]
              .value_counts().unstack(fill_value=0)
              .reindex(index=order_display, columns=CONCLUSION_ORDER, fill_value=0))
        n_by     = df_bar.groupby("method_display").size().reindex(order_display, fill_value=0)
        bottoms  = np.zeros(len(order_display))
        x        = np.arange(len(order_display))
        colors   = [CONCLUSION_PALETTE[k] for k in CONCLUSION_ORDER]

        for outcome, color in zip(CONCLUSION_ORDER, colors):
            vals = ct[outcome].values if outcome in ct.columns else np.zeros(len(order_display))
            bars = ax_bar.bar(x, vals, bottom=bottoms, color=color,
                              label=CONCLUSION_EN[outcome], edgecolor="white", width=0.55)
            for i, (bar, v) in enumerate(zip(bars, vals)):
                if v > 0:
                    pct = v / n_by.iloc[i] * 100
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, bottoms[i] + v / 2,
                                f"{v}\n({pct:.0f}%)", ha="center", va="center",
                                fontsize=11, color="white", fontweight="bold")
            bottoms += vals

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"{g}\n(n={n_by[g]})" for g in order_display])
        ax_bar.set_ylabel("Count")
        ax_bar.legend(loc="upper right", title="")
        ax_bar.set_ylim(0, n_by.max() + 5)
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.grid(axis="y", alpha=0.25)

        ax_bar.text(-0.08, 1.04, "(b)", transform=ax_bar.transAxes,
                    fontsize=14, fontweight="bold", va="bottom")

        plt.tight_layout()
        plt.savefig(PLOT_DIR / "16_conclusion_panel.png", dpi=150,
                    facecolor="white", bbox_inches="tight")
        plt.close()
    print("  Saved 16_conclusion_panel.png")


# ---------------------------------------------------------------------------
# Analysis 2: Pro vs Self+App vs Other+App  (plots 12–14)
# ---------------------------------------------------------------------------

def print_threeway_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("  ANALYSIS 2 — PRO vs SELF+APP vs OTHER+APP")
    print("=" * 60)
    for method in METHOD_THREE_ORDER:
        sub = df[df["method"] == method]
        print(f"\n{method} (n={len(sub)}):")
        counts = sub["conclusion"].value_counts().reindex(CONCLUSION_ORDER, fill_value=0)
        for c, n in counts.items():
            print(f"  {c:<30s}: {n:2d}  ({n/len(sub)*100:.0f}%)")
        print(f"  Mean |error| (all electrodes): "
              f"{np.nanmean(sub[[f'{e}_true_abs' for e in ELEC_ORDER]].values):.3f} cm")

    print("\n--- Per-electrode mean |error| (cm) ---")
    print(f"  {'Electrode':<15} {'Pro':>8} {'Self+App':>10} {'Other+App':>11}")
    for e in ELEC_ORDER:
        vals = {m: df[df["method"] == m][f"{e}_true_abs"].mean() for m in METHOD_THREE_ORDER}
        print(f"  {ELECTRODE_LABELS[e]:<15} "
              f"{vals['Pro']:>8.3f} {vals['Self + App']:>10.3f} {vals['Other + App']:>11.3f}")
    print()


def plot_threeway_conclusion(df: pd.DataFrame) -> None:
    """Plot 12."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _conclusion_bars(ax, df, "method", METHOD_THREE_ORDER,
                     "Placement Outcome: Pro vs. Self+App vs. Other+App")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "12_threeway_conclusion.png", dpi=150)
    plt.close()
    print("  Saved 12_threeway_conclusion.png")


def plot_threeway_electrode(df: pd.DataFrame) -> None:
    """Plot 13 — |error| and % deviation, 3-way."""
    long = _build_long_by_method(df, "method")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, ycol, title in [
        (axes[0], "|Error| (cm)",  "|Error| — 3-way Comparison"),
        (axes[1], "Deviation (%)", "Signed % Deviation — 3-way Comparison"),
    ]:
        sns.boxplot(data=long, x="Electrode", y=ycol,
                    hue="method", hue_order=METHOD_THREE_ORDER,
                    order=LABEL_ORDER, ax=ax, linewidth=1.1,
                    palette=METHOD_THREE_PAL,
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("")
        ax.legend(title="", loc="upper right", fontsize=8.5)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "13_threeway_electrodes.png", dpi=150)
    plt.close()
    print("  Saved 13_threeway_electrodes.png")


def plot_threeway_means(df: pd.DataFrame) -> None:
    """Plot 14 — per-electrode mean |error| ± 95 % CI with individual points."""
    with plt.rc_context(STYLE_RC):
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(ELEC_ORDER))
        offsets = {"Pro": -0.28, "Self + App": 0.0, "Other + App": 0.28}
        width = 0.22
        for method, offset in offsets.items():
            sub   = df[df["method"] == method]
            label = METHOD_DISPLAY.get(method, method)
            means, ses, all_vals = [], [], []
            for e in ELEC_ORDER:
                v = sub[f"{e}_true_abs"].dropna()
                means.append(v.mean())
                ses.append(v.sem())
                all_vals.append(v.values)
            ax.bar(x + offset, means, width,
                   color=METHOD_THREE_PAL[method], label=f"{label} (n={len(sub)})",
                   alpha=0.80, edgecolor="white")
            ax.errorbar(x + offset, means, yerr=[1.96 * s for s in ses],
                        fmt="none", color="black", capsize=3, capthick=1.2, lw=1.2)
            for xi, vals in zip(x + offset, all_vals):
                jitter = np.random.default_rng(42).uniform(-0.06, 0.06, size=len(vals))
                ax.scatter(xi + jitter, vals, color=METHOD_THREE_PAL[method],
                           s=22, alpha=0.7, edgecolors="white", linewidth=0.4, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_ORDER, rotation=20, ha="right")
        ax.set_ylabel("|Error| (cm)")
        ax.legend(loc="upper right")
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "14_threeway_means.png", dpi=150)
        plt.close()
    print("  Saved 14_threeway_means.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nLoading data...")
    meta    = _parse_metadata()
    results = _parse_results()
    trial_order = _parse_trial_order()

    df_full = (results
               .merge(trial_order, on=["subject_id", "trial"], how="left")
               .merge(meta, on="subject_id", how="left"))

    # Drop subjects who did not complete both trials
    dev_cols = [f"{e}_dev" for e in ELEC_KEYS]
    incomplete = (df_full.groupby("subject_id", group_keys=False)
                  .apply(lambda g: g[dev_cols].isna().all(axis=1).any()))
    excluded = incomplete[incomplete].index.tolist()
    if excluded:
        print(f"Excluding subject(s) with incomplete trials: {excluded}")
    df   = _add_true_abs(df_full[~df_full["subject_id"].isin(excluded)].copy())
    meta = meta[~meta["subject_id"].isin(excluded)].copy()

    print_summary(df, meta)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 100})

    print("Generating overview plots...")
    plot_signed_deviation_boxplot(df)
    plot_pct_deviation_violin(df)
    plot_conclusion_distribution(df)
    plot_heatmap(df)
    plot_sex_comparison(df)
    plot_hair_vs_outcome(df)
    plot_age_analysis(df, meta)
    plot_trial_comparison(df)

    print("\nGenerating Analysis 1 — Pro vs App...")
    print_kappa_analysis(df)
    print_method_summary(df)
    plot_pro_vs_app_conclusion(df)
    plot_pro_vs_app_electrode(df)
    plot_pro_vs_app_paired(df)
    plot_pro_vs_app_confusion(df)
    plot_conclusion_panel(df)

    print("\nGenerating Analysis 2 — Pro vs Self+App vs Other+App...")
    print_threeway_summary(df)
    plot_threeway_conclusion(df)
    plot_threeway_electrode(df)
    plot_threeway_means(df)

    print(f"\nAll plots saved to: {PLOT_DIR}\n")


if __name__ == "__main__":
    main()
