#!/usr/bin/env python3
"""Publication-ready figures for the clinical paper.

`analyze_study.py` writes the screen-resolution exploratory figures. This
script writes the composites that appear in the manuscript itself, at 300 dpi:

  figure2_combined.png   Figure 2   — (a) Expert vs App-guided outcome
                                      confusion matrix, (b) Bland-Altman on
                                      per-participant mean absolute error
  figure3_threearm.png   Figure S3  — (a) outcome distribution and
                                      (b) per-electrode error, resolving the
                                      App arm into Self- and Helper-guided
  17_bland_altman.png               — the Bland-Altman panel standalone
  18_head_diagram.png               — 10-20 measurement schematic (no data)

Reads only the CSVs in `data/`; writes to `outputs/`. The data loader here is
deliberately independent of `analyze_study.py` so the manuscript figures and
the reported statistics are derived by two separate code paths from the same
published inputs — if they disagree, one of them is wrong.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import stats

warnings.filterwarnings("ignore")

HERE     = Path(__file__).parent
DATA_DIR = HERE / "data"
PLOT_DIR = HERE / "outputs"
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
METHOD_DISPLAY = {"Pro": "Expert", "Self + App": "Self-guided", "Other + App": "Helper-guided"}
METHOD_THREE_ORDER = ["Pro", "Self + App", "Other + App"]
METHOD_THREE_PAL   = {"Pro": "#0077BB", "Self + App": "#EE7733", "Other + App": "#009988"}

STYLE_RC = {
    "font.family":       "STIXGeneral",
    "font.size":         13,
    "axes.titleweight":  "normal",
    "axes.labelsize":    13,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   11,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data() -> pd.DataFrame:
    """Trial-level frame: deviations, absolute errors, arm, and outcome.

    ``float_precision="round_trip"`` is required — pandas' default CSV float
    parser is fast but not correctly rounded, and the last-bit error changes
    the reported means in the final digit. See the note in the root README.
    """
    df_results = pd.read_csv(DATA_DIR / "measurements.csv", float_precision="round_trip")
    df_trial = pd.read_csv(DATA_DIR / "trial_order.csv", float_precision="round_trip")

    df = df_results.merge(df_trial, on=["subject_id", "trial"], how="left")
    for e in ELEC_KEYS:
        df[f"{e}_true_abs"] = df[f"{e}_dev"].abs()
    df["mean_err"] = df[[f"{e}_true_abs" for e in ELEC_KEYS]].mean(axis=1)

    # A participant with any trial missing every measurement is dropped whole,
    # not per-trial, so the cross-over stays balanced. n = 31 -> 30.
    dev_cols = [f"{e}_dev" for e in ELEC_KEYS]
    incomplete = (
        df.groupby("subject_id", group_keys=False)
        .apply(lambda g: g[dev_cols].isna().all(axis=1).any())
    )
    excluded = incomplete[incomplete].index.tolist()
    return df[~df["subject_id"].isin(excluded)].copy()


# ---------------------------------------------------------------------------
# Bland-Altman, standalone
# ---------------------------------------------------------------------------

def plot_bland_altman(df: pd.DataFrame) -> None:
    """Bland-Altman: Expert vs App-guided mean |error|, per participant."""
    pro = df[df["method_simple"] == "Pro"].set_index("subject_id")["mean_err"]
    app = df[df["method_simple"] == "App"].set_index("subject_id")["mean_err"]
    paired = pd.concat([pro.rename("pro"), app.rename("app")], axis=1).dropna()

    mean_vals = (paired["pro"] + paired["app"]) / 2
    diff_vals = paired["app"] - paired["pro"]

    mean_diff  = diff_vals.mean()
    sd_diff    = diff_vals.std()
    loa_upper  = mean_diff + 1.96 * sd_diff
    loa_lower  = mean_diff - 1.96 * sd_diff
    n          = len(paired)

    # 95 % CI on the LoAs (assuming normality)
    se_loa = np.sqrt(3 * sd_diff**2 / n)
    ci_loa = 1.96 * se_loa

    with plt.rc_context(STYLE_RC):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        ax.scatter(mean_vals, diff_vals, color="#2166AC", s=55, alpha=0.85,
                   edgecolors="white", linewidth=0.6, zorder=3, label="Participant")

        xmin, xmax = mean_vals.min() - 0.05, mean_vals.max() + 0.05
        ax.axhline(mean_diff,  color="#333333", lw=1.8,  ls="-",  zorder=2)
        ax.axhline(loa_upper,  color="#CC3311", lw=1.4,  ls="--", zorder=2)
        ax.axhline(loa_lower,  color="#CC3311", lw=1.4,  ls="--", zorder=2)
        ax.axhline(0,          color="gray",   lw=0.8,  ls=":",  alpha=0.5, zorder=1)

        # Shaded LoA bands (95 % CI of LoA)
        ax.fill_between([xmin, xmax],
                         [loa_upper - ci_loa] * 2, [loa_upper + ci_loa] * 2,
                         color="#CC3311", alpha=0.10, zorder=0)
        ax.fill_between([xmin, xmax],
                         [loa_lower - ci_loa] * 2, [loa_lower + ci_loa] * 2,
                         color="#CC3311", alpha=0.10, zorder=0)
        ax.fill_between([xmin, xmax],
                         [mean_diff - se_loa] * 2, [mean_diff + se_loa] * 2,
                         color="#333333", alpha=0.12, zorder=0)

        ax.text(xmax + 0.005, mean_diff,
                f"Mean\n{mean_diff:+.3f} cm", va="center", ha="left", fontsize=10)
        ax.text(xmax + 0.005, loa_upper,
                f"+1.96 SD\n{loa_upper:+.3f} cm", va="center", ha="left", fontsize=10,
                color="#CC3311")
        ax.text(xmax + 0.005, loa_lower,
                f"−1.96 SD\n{loa_lower:+.3f} cm", va="center", ha="left", fontsize=10,
                color="#CC3311")

        ax.set_xlim(xmin, xmax + 0.22)
        ax.set_xlabel("Mean of Expert and App-Guided |Error| (cm)")
        ax.set_ylabel("App-Guided minus Expert |Error| (cm)")
        ax.spines[["top", "right"]].set_visible(False)

        # Pearson r for trend test
        r, p_r = stats.pearsonr(mean_vals, diff_vals)
        p_label = f"p = {p_r:.3f}" if p_r >= 0.001 else "p < 0.001"
        ax.text(0.03, 0.97,
                f"n = {n} participants\nProportion bias: r = {r:.2f}, {p_label}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=0.8))

        plt.tight_layout()
        plt.savefig(PLOT_DIR / "17_bland_altman.png", dpi=300,
                    facecolor="white", bbox_inches="tight")
        plt.close()
    print("  Saved 17_bland_altman.png (300 dpi)")



# ---------------------------------------------------------------------------
# Head diagram — 10-20 measurement positions (schematic, no data)
# ---------------------------------------------------------------------------

def plot_head_diagram() -> None:
    """
    Publication-quality schematic of the 10-20 measurement positions.

    Panel (a) — Top-down (axial) view:
        Shows all 6 electrode positions (Fp1/Fp2, T7/T8, O1/O2) on a head
        outline with measurement arcs.

    Panel (b) — Sagittal (side) view:
        Shows the anterior-posterior arc (Nasion-Cz-Inion) and the
        Fp / O electrode positions along it.
    """
    ELEC_COLOR = "#D32F2F"
    ARROW_COLOR = "#1565C0"
    HEAD_COLOR = "#444444"
    LABEL_PAD = 0.04

    with plt.rc_context({**STYLE_RC, "font.size": 11, "axes.labelsize": 11}):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 6.5),
                                          facecolor="white")
        for ax in (ax_a, ax_b):
            ax.set_aspect("equal")
            ax.axis("off")

        # ── Panel (a): Top-down view ──────────────────────────────────────────
        cx, cy = 0.0, 0.0
        rx, ry = 0.42, 0.52   # head semi-axes

        # Head outline
        head = Ellipse((cx, cy), 2 * rx, 2 * ry,
                       fill=True, facecolor="#FAF9F6", edgecolor=HEAD_COLOR,
                       linewidth=2.0, zorder=2)
        ax_a.add_patch(head)

        # Nose at top
        nose_y = cy + ry
        for dx in [-0.06, 0.06]:
            ax_a.plot([cx + dx, cx], [nose_y, nose_y + 0.09],
                      color=HEAD_COLOR, lw=1.6, zorder=3)

        # Reference lines (thin dashed)
        ax_a.plot([-rx * 0.95, rx * 0.95], [cy, cy], color="#BBBBBB", lw=0.8,
                  ls="--", zorder=1)
        ax_a.plot([cx, cx], [-ry * 0.92, ry * 0.92], color="#BBBBBB", lw=0.8,
                  ls="--", zorder=1)

        # Electrode positions (top-down, approximate 10-20 fractions)
        # Fp1 and Fp2 are at 10% from nasion, 10% from midline
        FP_Y = cy + ry * 0.65
        FP_X = 0.16
        O_Y  = cy - ry * 0.65
        O_X  = 0.16
        T_X  = rx
        T_Y  = cy

        electrodes = {
            "Fp1": (-FP_X,  FP_Y),
            "Fp2": ( FP_X,  FP_Y),
            "T7":  (-T_X,   T_Y),
            "T8":  ( T_X,   T_Y),
            "O1":  (-O_X,   O_Y),
            "O2":  ( O_X,   O_Y),
            "Cz":  (0.0,    0.0),
        }

        for name, (ex, ey) in electrodes.items():
            r = 0.04 if name != "Cz" else 0.03
            fc = ELEC_COLOR if name != "Cz" else "#888888"
            dot = plt.Circle((ex, ey), r, color=fc, zorder=5)
            ax_a.add_patch(dot)
            offsets = {
                "Fp1": (-LABEL_PAD - 0.04, 0.02), "Fp2": ( LABEL_PAD, 0.02),
                "T7":  (-LABEL_PAD - 0.08, 0.0),  "T8":  ( LABEL_PAD, 0.0),
                "O1":  (-LABEL_PAD - 0.04, -0.04),"O2":  ( LABEL_PAD, -0.04),
                "Cz":  (0.04, 0.02),
            }
            ox, oy = offsets[name]
            ax_a.text(ex + ox, ey + oy, name,
                      fontsize=11, fontweight="bold", color=fc,
                      va="center", ha="left" if ox >= 0 else "right", zorder=6)

        # Transverse arc annotation (T7–Cz–T8)
        ax_a.annotate("", xy=( T_X, T_Y), xytext=(-T_X, T_Y),
                      arrowprops=dict(arrowstyle="<->", color=ARROW_COLOR,
                                      lw=1.5,
                                      connectionstyle="arc3,rad=-0.35"))
        ax_a.text(cx, T_Y + 0.10, "Transverse arc\n(PA–Cz–PA)",
                  ha="center", va="bottom", fontsize=9.5, color=ARROW_COLOR)

        # AP arc annotation (Fp–Cz–O, left side only)
        ax_a.annotate("", xy=(-FP_X, FP_Y), xytext=(-O_X, O_Y),
                      arrowprops=dict(arrowstyle="<->", color="#009944",
                                      lw=1.5,
                                      connectionstyle="arc3,rad=0.55"))
        ax_a.text(-rx - 0.08, cy, "A-P arc\n(Nasion–Cz–Inion)",
                  ha="right", va="center", fontsize=9.5, color="#009944",
                  rotation=90)

        # Lateral offset arrows for Fp1 (lat)
        ax_a.annotate("", xy=(-FP_X, FP_Y), xytext=(0, FP_Y),
                      arrowprops=dict(arrowstyle="<->", color="#7B1FA2", lw=1.3))
        ax_a.text(-FP_X / 2, FP_Y + 0.05, "lat", ha="center", va="bottom",
                  fontsize=9, color="#7B1FA2")

        # AP offset for Fp1
        # Find the arc position at Fp1's lateral distance from midline
        ax_a.annotate("", xy=(-FP_X, FP_Y), xytext=(-FP_X, FP_Y + ry * 0.28),
                      arrowprops=dict(arrowstyle="<->", color="#7B1FA2", lw=1.3))
        ax_a.text(-FP_X - 0.05, FP_Y + ry * 0.14, "A-P", ha="right", va="center",
                  fontsize=9, color="#7B1FA2")

        ax_a.set_xlim(-rx - 0.28, rx + 0.28)
        ax_a.set_ylim(-ry - 0.22, ry + 0.28)
        ax_a.text(cx, ry + 0.22, "(a) Top-down view",
                  ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax_a.text(cx, ry + 0.08, "Nasion (front)",
                  ha="center", va="bottom", fontsize=9, style="italic", color="#888888")
        ax_a.text(cx, -ry - 0.17, "Inion (back)",
                  ha="center", va="top", fontsize=9, style="italic", color="#888888")

        # ── Panel (b): Sagittal (left side) view ─────────────────────────────
        # Ellipse: wider in x (A-P) than y (superior-inferior)
        cx2, cy2 = 0.0, 0.0
        rx2, ry2 = 0.55, 0.45

        head2 = Ellipse((cx2, cy2), 2 * rx2, 2 * ry2,
                        fill=True, facecolor="#FAF9F6", edgecolor=HEAD_COLOR,
                        linewidth=2.0, zorder=2)
        ax_b.add_patch(head2)

        # Nasion (front) and Inion (back) anchor points
        nasion = (-rx2, 0.0)
        inion  = ( rx2, 0.0)

        # A-P arc (dashed line along the arc, shown as curved path through Cz)
        t_arc = np.linspace(np.pi, 0, 200)
        arc2_x = rx2 * np.cos(t_arc)
        arc2_y = ry2 * np.sin(t_arc)
        ax_b.plot(arc2_x, arc2_y, color="#009944", lw=1.5, ls="--", zorder=3,
                  label="A-P arc (Nasion–Cz–Inion)")

        # Electrode positions along the arc (approximate arc-length fractions)
        # Fp1: 10% from Nasion along the arc
        # O1:  10% from Inion along the arc
        def arc_pos(frac: float) -> tuple[float, float]:
            theta = np.pi * (1 - frac)
            return rx2 * np.cos(theta), ry2 * np.sin(theta)

        fp1_pos = arc_pos(0.10)   # 10% from Nasion
        o1_pos  = arc_pos(0.90)   # 10% from Inion
        cz_arc  = arc_pos(0.50)

        for pos, name in [(fp1_pos, "Fp1"), (o1_pos, "O1"), (cz_arc, "Cz")]:
            fc = ELEC_COLOR if name != "Cz" else "#888888"
            r = 0.04 if name != "Cz" else 0.03
            dot = plt.Circle(pos, r, color=fc, zorder=5)
            ax_b.add_patch(dot)
            ox = -0.07 if name == "Fp1" else (0.07 if name == "O1" else 0.06)
            oy = 0.02
            ax_b.text(pos[0] + ox, pos[1] + oy, name,
                      fontsize=11, fontweight="bold", color=fc,
                      ha="left" if ox > 0 else "right", va="center", zorder=6)

        # T7 on the temporal region: preauricular point is roughly at the
        # equatorial level, slightly anterior of center in the sagittal view.
        t7_pos = (-rx2 * 0.18, -ry2 * 0.05)
        # Snap to the head surface
        theta_t7 = np.arctan2(t7_pos[1] / ry2, t7_pos[0] / rx2)
        t7_pos = (rx2 * np.cos(theta_t7), ry2 * np.sin(theta_t7))
        dot_t7 = plt.Circle(t7_pos, 0.04, color=ELEC_COLOR, zorder=5)
        ax_b.add_patch(dot_t7)
        ax_b.text(t7_pos[0] - 0.06, t7_pos[1] - 0.10, "T7\n(preauricular)",
                  fontsize=10, fontweight="bold", color=ELEC_COLOR,
                  ha="center", va="top", zorder=6)

        # Nasion and Inion markers
        for pos, label, ha in [(nasion, "Nasion", "right"), (inion, "Inion", "left")]:
            dot = plt.Circle(pos, 0.03, color="#888888", zorder=5)
            ax_b.add_patch(dot)
            ox2 = -0.05 if ha == "right" else 0.05
            ax_b.text(pos[0] + ox2, pos[1], label,
                      fontsize=9.5, style="italic", color="#666666",
                      ha=ha, va="center", zorder=6)

        # A-P offset annotation for Fp1 (arc from Nasion to Fp1)
        nasion_arc = (-rx2, 0.0)
        ax_b.annotate("", xy=fp1_pos, xytext=nasion_arc,
                      arrowprops=dict(arrowstyle="<->", color=ARROW_COLOR,
                                      lw=1.5,
                                      connectionstyle="arc3,rad=-0.35"),
                      zorder=4)
        ax_b.text(-rx2 * 0.80, 0.30, "10 % of\nA-P arc",
                  ha="center", va="bottom", fontsize=9.5, color=ARROW_COLOR)

        # A-P offset annotation for O1 (arc from Inion to O1)
        inion_arc = ( rx2, 0.0)
        ax_b.annotate("", xy=o1_pos, xytext=inion_arc,
                      arrowprops=dict(arrowstyle="<->", color=ARROW_COLOR,
                                      lw=1.5,
                                      connectionstyle="arc3,rad=0.35"),
                      zorder=4)
        ax_b.text(rx2 * 0.80, 0.30, "10 % of\nA-P arc",
                  ha="center", va="bottom", fontsize=9.5, color=ARROW_COLOR)

        # T7 transverse arc annotation (arc from T7 to Cz)
        ax_b.annotate("", xy=t7_pos, xytext=cz_arc,
                      arrowprops=dict(arrowstyle="<->", color="#009944",
                                      lw=1.4,
                                      connectionstyle="arc3,rad=0.5"),
                      zorder=4)
        ax_b.text(-rx2 * 0.20, ry2 * 0.72, "10 % of\ntransverse arc",
                  ha="center", va="center", fontsize=9, color="#009944")

        ax_b.set_xlim(-rx2 - 0.22, rx2 + 0.22)
        ax_b.set_ylim(-ry2 - 0.20, ry2 + 0.22)
        ax_b.text(cx2, ry2 + 0.17, "(b) Sagittal view (left hemisphere)",
                  ha="center", va="bottom", fontsize=12, fontweight="bold")

        # Shared legend
        handles = [
            mpatches.Patch(color=ELEC_COLOR, label="Measured electrode"),
            mpatches.Patch(color="#888888",  label="Reference point"),
            plt.Line2D([0], [0], color="#009944", lw=1.5, ls="--",
                       label="A-P arc / transverse arc"),
            plt.Line2D([0], [0], color=ARROW_COLOR,  lw=1.5,
                       label="10 % arc measurement"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   fontsize=10, frameon=True, framealpha=1.0,
                   edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.0))

        plt.tight_layout(rect=[0, 0.07, 1, 1])
        plt.savefig(PLOT_DIR / "18_head_diagram.png", dpi=300,
                    facecolor="white", bbox_inches="tight")
        plt.close()
    print("  Saved 18_head_diagram.png (300 dpi)")


# ---------------------------------------------------------------------------
# Figure 2 — confusion matrix + Bland-Altman combined panel
# ---------------------------------------------------------------------------

def plot_figure2(df: pd.DataFrame) -> None:
    """
    Two-panel publication figure:
      (a) Confusion matrix  — Expert vs App-guided placement outcome
      (b) Bland-Altman      — Expert vs App-guided mean |error|, per participant
    Both panels share the same typeface, spine treatment, and colour family.
    """
    # ── shared palette (mirrors analyze_study.py 15_pro_vs_app_confusion_matrix) ──
    AGREE_HI  = "#2166AC"
    AGREE_MID = "#74B2D8"
    PRO_HI    = "#FDBF6F"
    PRO_LO    = "#FEE4C4"
    PRO_INK   = "#7A3500"
    APP_FILL  = "#B2E0D6"
    APP_INK   = "#1A5E50"
    ZERO_FILL = "#F7F7F7"
    CELL_EDGE = "#DDDDDD"
    WHITE     = "#FFFFFF"
    GHOST_INK = "#AAAAAA"
    INK       = "#2A2A2A"

    # ── confusion matrix data ─────────────────────────────────────────────────
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
    max_diag = max((conf[i, i] for i in range(k)), default=1) or 1

    def cell_style(i: int, j: int) -> tuple[str, str]:
        n = conf[i, j]
        if i == j:
            return (ZERO_FILL, GHOST_INK) if n == 0 else \
                   ((AGREE_HI, WHITE) if n >= 0.7 * max_diag else (AGREE_MID, WHITE))
        if i < j:
            return (ZERO_FILL, GHOST_INK) if n == 0 else \
                   ((PRO_HI, PRO_INK) if n >= 5 else (PRO_LO, PRO_INK))
        return (ZERO_FILL, GHOST_INK) if n == 0 else (APP_FILL, APP_INK)

    # ── Bland-Altman data ─────────────────────────────────────────────────────
    pro_err = df[df["method_simple"] == "Pro"].set_index("subject_id")["mean_err"]
    app_err = df[df["method_simple"] == "App"].set_index("subject_id")["mean_err"]
    paired  = pd.concat([pro_err.rename("pro"), app_err.rename("app")], axis=1).dropna()
    mean_v  = (paired["pro"] + paired["app"]) / 2
    diff_v  = paired["app"] - paired["pro"]
    mean_d  = diff_v.mean()
    sd_d    = diff_v.std()
    loa_up  = mean_d + 1.96 * sd_d
    loa_lo  = mean_d - 1.96 * sd_d

    cats_en = ["Optimal", "Usable", "Incorrect"]

    rc = {
        "font.family":       "STIXGeneral",
        "font.size":         12,
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         False,
    }

    with plt.rc_context(rc):
        from matplotlib.patches import Rectangle, Patch

        # ── Figure geometry (inches) ──────────────────────────────────────────
        # Both panels share the same y0 and height → guaranteed equal height.
        # Confusion matrix width = height (square cells because xlim == ylim span).
        fig_w, fig_h = 13.0, 6.2
        left_in  = 0.55   # y-label of (a)
        top_in   = 0.85   # top x-axis title + ticks + panel labels
        bot_in   = 1.45   # legend + x-label + column marginals
        gap_in   = 1.30   # row marginals + (b) y-ticks/label
        right_in = 0.15

        ph = fig_h - top_in - bot_in          # panel height = 3.90"
        cw = ph                                # confusion matrix: square
        bw = fig_w - left_in - cw - gap_in - right_in  # BA width = 7.05"

        def nx(v): return v / fig_w
        def ny(v): return v / fig_h

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        ax_cm = fig.add_axes([nx(left_in),              ny(bot_in), nx(cw), ny(ph)])
        ax_ba = fig.add_axes([nx(left_in + cw + gap_in), ny(bot_in), nx(bw), ny(ph)])

        # ════════════════════════════════════════════════════════════════════
        # Panel (a) — Confusion matrix
        # ════════════════════════════════════════════════════════════════════
        ax_cm.set_facecolor("white")

        for i in range(k):
            for j in range(k):
                fc, tc = cell_style(i, j)
                n = conf[i, j]
                ax_cm.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor=fc, edgecolor=CELL_EDGE, linewidth=0.6, zorder=2))
                ax_cm.text(j, i - 0.08, str(n),
                           ha="center", va="center",
                           fontsize=28, color=tc,
                           fontweight="bold" if n > 0 else "normal", zorder=4)
                if n > 0:
                    ax_cm.text(j, i + 0.30, f"{n / n_pairs * 100:.0f}%",
                               ha="center", va="center", fontsize=9.5,
                               color=tc, alpha=0.80, zorder=4)

        # Outer border
        ax_cm.add_patch(Rectangle((-0.5, -0.5), k, k,
                                   facecolor="none", edgecolor="#999999",
                                   linewidth=0.8, zorder=5))

        ax_cm.set_xticks(range(k))
        ax_cm.set_yticks(range(k))
        ax_cm.set_xticklabels(cats_en, fontsize=10)
        ax_cm.set_yticklabels(cats_en, fontsize=10)
        ax_cm.xaxis.set_ticks_position("top")
        ax_cm.xaxis.set_label_position("top")
        ax_cm.set_xlabel("App-Guided Placement", fontsize=11, labelpad=10)
        ax_cm.set_ylabel("Expert Placement", fontsize=11, labelpad=10)
        # Equal spans → cells stay square when the axes box is made square below
        ax_cm.set_xlim(-0.5, k - 0.5 + 0.50)   # span = 3.50
        ax_cm.set_ylim(k - 0.5 + 0.50, -0.5)    # span = 3.50 (inverted)
        for spine in ax_cm.spines.values():
            spine.set_visible(False)
        ax_cm.tick_params(length=0, colors="#444444")

        # Marginal totals
        for i in range(k):
            ax_cm.text(k - 0.5 + 0.06, i, f"n = {row_m[i]}",
                       ha="left", va="center", fontsize=8.5, color="#999999")
        for j in range(k):
            ax_cm.text(j, k - 0.5 + 0.14, f"n = {col_m[j]}",
                       ha="center", va="top", fontsize=8.5, color="#999999")

        # Panel labels added after tight_layout (see below)

        # ════════════════════════════════════════════════════════════════════
        # Panel (b) — Bland-Altman
        # ════════════════════════════════════════════════════════════════════
        ax_ba = fig.axes[1]  # already created by add_axes above

        xlo = mean_v.min() - 0.08
        xhi = mean_v.max() + 0.08

        # Zero reference — very faint
        ax_ba.axhline(0, color="#CCCCCC", lw=0.8, ls=":", zorder=1)

        # LoA and mean lines
        LINE_KW = dict(lw=1.4, zorder=2)
        ax_ba.axhline(mean_d, color=INK,      ls="-",  **LINE_KW)
        ax_ba.axhline(loa_up, color=AGREE_HI, ls="--", **LINE_KW)
        ax_ba.axhline(loa_lo, color=AGREE_HI, ls="--", **LINE_KW)

        # ±0.5 cm reference lines
        ax_ba.axhline(0.5,  color="black", ls="--", **LINE_KW)
        ax_ba.axhline(-0.5, color="black", ls="--", **LINE_KW)

        # Scatter — same deep blue as agreement cells
        ax_ba.scatter(mean_v, diff_v,
                      color=AGREE_HI, s=48, alpha=0.85,
                      edgecolors="white", linewidth=0.6, zorder=3)

        # Inline labels: right-aligned, just inside the right edge,
        # offset above/below the line so they never sit on top of it.
        label_x = xhi - 0.02
        for y_val, txt, color, va_off, dy in [
            (loa_up, f"+1.96 SD  {loa_up:+.2f} cm", AGREE_HI, "top",    -0.015),
            (mean_d, f"Mean  {mean_d:+.3f} cm",      INK,      "bottom", +0.015),
            (loa_lo, f"−1.96 SD  {loa_lo:+.2f} cm", AGREE_HI, "top",    -0.015),
        ]:
            ax_ba.text(label_x, y_val + dy, txt,
                       va=va_off, ha="right", fontsize=8.5, color=color, zorder=5)

        # ±0.5 cm labels: left-aligned, near the left edge (right edge is
        # crowded by the SD/mean labels and, at +0.5, by the outlier point).
        # Both labels sit above their line so they don't collide with the
        # x-axis (for −0.5 cm) or the outlier point (for +0.5 cm).
        label_x2 = xlo + 0.02
        for y_val, txt in [
            (0.5,  "+0.5 cm"),
            (-0.5, "−0.5 cm"),
        ]:
            ax_ba.text(label_x2, y_val + 0.015, txt,
                       va="bottom", ha="left", fontsize=8.5, color="black", zorder=5)

        ax_ba.set_xlim(xlo, xhi + 0.04)
        # y-range: tight but show the outlier; clip top pad so legend/header fit
        ypad_lo = 0.10
        ypad_hi = 0.08
        ax_ba.set_ylim(diff_v.min() - ypad_lo, diff_v.max() + ypad_hi)

        # Light shading between the LoA lines (drawn after xlim is set)
        ax_ba.fill_between([xlo, xhi + 0.04], loa_lo, loa_up,
                           color=AGREE_HI, alpha=0.07, zorder=0)

        ax_ba.set_xlabel(r"Mean absolute error (cm)", fontsize=11)
        ax_ba.set_ylabel(r"App $-$ Expert (cm)", fontsize=11)

        ax_ba.spines["top"].set_visible(False)
        ax_ba.spines["right"].set_visible(False)
        ax_ba.spines["left"].set_color("#BBBBBB")
        ax_ba.spines["bottom"].set_color("#BBBBBB")
        ax_ba.tick_params(colors="#444444")

        # Panel labels added after tight_layout (see below)

        # ════════════════════════════════════════════════════════════════════
        # Legend anchored under panel (a) only — keeps it away from (b) labels
        # ════════════════════════════════════════════════════════════════════
        legend_handles = [
            Patch(facecolor=AGREE_HI,  edgecolor=CELL_EDGE, lw=0.6, label="Agreement"),
            Patch(facecolor=PRO_HI,    edgecolor=CELL_EDGE, lw=0.6, label="Expert > App"),
            Patch(facecolor=APP_FILL,  edgecolor=CELL_EDGE, lw=0.6, label="App > Expert"),
            Patch(facecolor=ZERO_FILL, edgecolor=CELL_EDGE, lw=0.6, label="No observations"),
        ]
        # Legend centred under panel (a)
        fig.legend(handles=legend_handles,
                   loc="center", bbox_to_anchor=(nx(left_in + cw / 2), ny(bot_in * 0.42)),
                   ncol=2, fontsize=9.5, frameon=True, framealpha=1.0,
                   edgecolor="#DDDDDD", handlelength=1.0, handleheight=0.90,
                   borderpad=0.5, columnspacing=1.0)

        # Panel labels — both at identical figure-space y, just above the axes top
        label_y = ny(bot_in + ph) + 0.01
        fig.text(nx(left_in),              label_y, "(a)",
                 fontsize=13, fontweight="bold", va="bottom", ha="left",
                 fontfamily="STIXGeneral")
        fig.text(nx(left_in + cw + gap_in), label_y, "(b)",
                 fontsize=13, fontweight="bold", va="bottom", ha="left",
                 fontfamily="STIXGeneral")

        plt.savefig(PLOT_DIR / "figure2_combined.png", dpi=300,
                    facecolor="white", bbox_inches="tight")
        plt.close()
    print("  Saved figure2_combined.png (300 dpi)")



# ---------------------------------------------------------------------------
# Supplementary Figure 3 — three-arm outcomes (a) + per-electrode error (b)
# ---------------------------------------------------------------------------

def plot_figure3(df: pd.DataFrame) -> None:
    order_display = ["Expert", "Self-guided", "Helper-guided"]

    with plt.rc_context(STYLE_RC):
        fig = plt.figure(figsize=(15, 5.6), facecolor="white")
        gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.5], wspace=0.18)
        ax_bar = fig.add_subplot(gs[0])
        ax_err = fig.add_subplot(gs[1])

        # ── (a) three-arm outcome stacked bars ────────────────────────────────
        df_bar = df.copy()
        df_bar["method_display"] = df_bar["method"].map(METHOD_DISPLAY)
        ct = (df_bar.groupby("method_display")["conclusion"]
              .value_counts().unstack(fill_value=0)
              .reindex(index=order_display, columns=CONCLUSION_ORDER, fill_value=0))
        n_by    = df_bar.groupby("method_display").size().reindex(order_display, fill_value=0)
        bottoms = np.zeros(len(order_display))
        x       = np.arange(len(order_display))

        MIN_INSIDE = 3.0   # segments shorter than this get an outside label
        for outcome in CONCLUSION_ORDER:
            vals = ct[outcome].values
            bars = ax_bar.bar(x, vals, bottom=bottoms, color=CONCLUSION_PALETTE[outcome],
                              label=CONCLUSION_EN[outcome], edgecolor="white", width=0.6)
            for i, (bar, v) in enumerate(zip(bars, vals)):
                if v <= 0:
                    continue
                pct = v / n_by.iloc[i] * 100
                cx  = bar.get_x() + bar.get_width() / 2
                cy  = bottoms[i] + v / 2
                if v >= MIN_INSIDE:
                    ax_bar.text(cx, cy, f"{v}\n({pct:.0f}%)", ha="center", va="center",
                                fontsize=11, color="white", fontweight="bold")
                else:
                    # Thin segment: annotate outside to the right with a leader line
                    x_txt = bar.get_x() + bar.get_width() + 0.28
                    ax_bar.annotate(
                        f"{v} ({pct:.0f}%)", xy=(cx + bar.get_width() / 2, cy),
                        xytext=(x_txt, cy), ha="left", va="center",
                        fontsize=10.5, color=CONCLUSION_PALETTE[outcome],
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=CONCLUSION_PALETTE[outcome],
                                        lw=1.0, shrinkA=0, shrinkB=2))
            bottoms += vals

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"{g}\n(n={n_by[g]})" for g in order_display])
        ax_bar.set_ylabel("Count")
        ax_bar.legend(loc="upper right", frameon=True, framealpha=1.0,
                      edgecolor="#CCCCCC")
        ax_bar.set_ylim(0, n_by.max() + 5)
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.grid(axis="y", alpha=0.25)
        ax_bar.text(-0.14, 1.02, "(a)", transform=ax_bar.transAxes,
                    fontsize=15, fontweight="bold", va="bottom")

        # ── (b) per-electrode mean |error| ± 95 % CI, grouped by arm ──────────
        xe      = np.arange(len(ELEC_ORDER))
        offsets = {"Pro": -0.28, "Self + App": 0.0, "Other + App": 0.28}
        width   = 0.22
        rng     = np.random.default_rng(42)

        for method, offset in offsets.items():
            sub   = df[df["method"] == method]
            label = METHOD_DISPLAY[method]
            means, ses, all_vals = [], [], []
            for e in ELEC_ORDER:
                v = sub[f"{e}_true_abs"].dropna()
                means.append(v.mean())
                ses.append(v.sem())
                all_vals.append(v.values)
            ax_err.bar(xe + offset, means, width,
                       color=METHOD_THREE_PAL[method], label=f"{label} (n={len(sub)})",
                       alpha=0.80, edgecolor="white")
            ax_err.errorbar(xe + offset, means, yerr=[1.96 * s for s in ses],
                            fmt="none", color="black", capsize=3, capthick=1.2, lw=1.2)
            for xi, vals in zip(xe + offset, all_vals):
                jitter = rng.uniform(-0.06, 0.06, size=len(vals))
                ax_err.scatter(xi + jitter, vals, color=METHOD_THREE_PAL[method],
                               s=20, alpha=0.7, edgecolors="white", linewidth=0.4, zorder=4)

        ax_err.set_xticks(xe)
        ax_err.set_xticklabels(LABEL_ORDER, rotation=20, ha="right")
        ax_err.set_ylabel("|Error| (cm)")
        ax_err.legend(loc="upper right", frameon=True, framealpha=1.0,
                      edgecolor="#CCCCCC")
        ax_err.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax_err.spines[["top", "right"]].set_visible(False)
        ax_err.grid(axis="y", alpha=0.25)
        ax_err.text(-0.055, 1.02, "(b)", transform=ax_err.transAxes,
                    fontsize=15, fontweight="bold", va="bottom")

        plt.tight_layout()
        plt.savefig(PLOT_DIR / "figure3_threearm.png", dpi=300,
                    facecolor="white", bbox_inches="tight")
        plt.close()
    print("  Saved figure3_threearm.png (300 dpi)")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nLoading data...")
    df = _load_data()
    print(f"  n = {df['subject_id'].nunique()} participants, {len(df)} trials")
    for m in METHOD_THREE_ORDER:
        print(f"    {METHOD_DISPLAY[m]:14s}: {len(df[df['method'] == m])} trials")

    print("\nGenerating Bland-Altman plot...")
    plot_bland_altman(df)

    print("\nGenerating Figure 2 (confusion matrix + Bland-Altman)...")
    plot_figure2(df)

    print("\nGenerating supplementary Figure 3 (three-arm)...")
    plot_figure3(df)

    print("\nGenerating head diagram (10-20 measurement schematic)...")
    plot_head_diagram()

    print(f"\nAll paper figures saved to: {PLOT_DIR}\n")


if __name__ == "__main__":
    main()
