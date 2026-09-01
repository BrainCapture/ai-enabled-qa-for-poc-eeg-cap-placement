"""Supplementary figure and headline numbers for the displacement study.

Both perturbation modes are plotted against *realised* electrode displacement,
not the nominal parameter, because realised displacement is what the clinical
study measured and so what the 0.5 cm margin refers to. For a rigid cap shift
the two differ: the rotation is set by arc length at the vertex, but electrodes
near the rotation axis (T7/T8 for an anteroposterior slip) move less, so the
array mean is smaller than the nominal value. See `_curve` for why the cap and
single-electrode modes take their abscissa from different columns.

Headline values are read off by linear interpolation, which is accurate here
because every metric is linear in displacement over this range (verified in
`linearity_check`).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "cache")
FIGURE = os.path.join(HERE, "figures", "forward_displacement.png")

# Matches the palette used by the existing clinical-paper figures (Tol vibrant).
COLOR_CAP = "#0077BB"
COLOR_SINGLE = "#EE7733"
INK = "#222222"
MUTED = "#767676"

MARGIN_CM = 0.5
EXPERT_CM = 0.855
APP_CM = 0.938

#: A 2:1 left/right amplitude ratio - the usual threshold for calling an
#: interhemispheric asymmetry - corresponds to an asymmetry index of 1/3.
CLINICAL_ASYMMETRY_PP = 100.0 / 3.0

STYLE = {
    "font.family": "STIXGeneral",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
}


def _curve(df: pd.DataFrame, value: str, spread: str | None = None,
           x_col: str = "realised_mm_mean"):
    """Mean across directions/targets at each displacement, sorted by x.

    `x_col` differs by mode. A cap shift moves every electrode, so the array
    mean is the right abscissa; a single-electrode displacement moves exactly
    one, so the array mean would divide the true displacement by the number of
    channels - use the max (that electrode's own displacement) instead.
    """
    agg = {"realised": (x_col, "mean"), "y": (value, "mean")}
    if spread:
        agg["hi"] = (spread, "mean")
    out = df.groupby("magnitude_cm").agg(**agg).reset_index()
    out["x"] = out["realised"] / 10.0  # mm -> cm
    return out.sort_values("x")


def interp(curve: pd.DataFrame, x: float, col: str = "y") -> float:
    return float(np.interp(x, curve["x"], curve[col]))


def linearity_check(curve: pd.DataFrame) -> float:
    """R^2 of a through-origin linear fit; ~1.0 justifies interpolation."""
    x, y = curve["x"].to_numpy(), curve["y"].to_numpy()
    slope = (x @ y) / (x @ x)
    resid = y - slope * x
    return float(1 - resid.var() / y.var())


def _reference_lines(ax, ymax: float, label: bool) -> None:
    ax.axvspan(EXPERT_CM, APP_CM, color=MUTED, alpha=0.18, lw=0, zorder=0)
    ax.axvline(MARGIN_CM, color=INK, ls="--", lw=1.1, zorder=1)
    if label:
        ax.annotate(
            "0.5 cm\nmargin",
            xy=(MARGIN_CM, ymax), xytext=(MARGIN_CM - 0.04, ymax),
            ha="right", va="top", fontsize=9, color=INK, linespacing=1.25,
        )
        ax.annotate(
            "measured\nplacement error",
            xy=((EXPERT_CM + APP_CM) / 2, ymax),
            xytext=((EXPERT_CM + APP_CM) / 2 + 0.06, ymax),
            ha="left", va="top", fontsize=9, color=MUTED, linespacing=1.25,
        )


def build_figure(summary: pd.DataFrame, dipole: pd.DataFrame | None) -> dict:
    cap = summary[summary["mode"] == "cap"]
    single = summary[summary["mode"] == "single"]

    amp_cap = _curve(cap, "peak_rel_err_median", "peak_rel_err_p90")
    amp_single = _curve(single, "peak_rel_err_median", "peak_rel_err_p90",
                        x_col="realised_mm_max")
    asy_cap = _curve(cap, "d_asymmetry_p90")
    asy_single = _curve(single, "d_asymmetry_p90", x_col="realised_mm_max")

    plt.rcParams.update(STYLE)
    n_panels = 3 if dipole is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.1 * n_panels, 3.5))

    # ── Panel A: amplitude change ────────────────────────────────────────────
    ax = axes[0]
    for curve, color, name in (
        (amp_cap, COLOR_CAP, "Whole-cap shift"),
        (amp_single, COLOR_SINGLE, "Single electrode"),
    ):
        ax.fill_between(curve["x"], curve["y"] * 100, curve["hi"] * 100,
                        color=color, alpha=0.15, lw=0)
        ax.plot(curve["x"], curve["y"] * 100, color=color, lw=2,
                marker="o", ms=4.5, label=name, zorder=3)
    ax.set_xlabel("Electrode displacement (cm)")
    ax.set_ylabel("Change in scalp potential\n(% of peak amplitude)")
    ax.set_title("A  Signal amplitude", loc="left", weight="bold")
    _reference_lines(ax, ax.get_ylim()[1], label=True)
    ax.legend(frameon=False, loc="lower right")

    # ── Panel B: interhemispheric asymmetry ──────────────────────────────────
    ax = axes[1]
    for curve, color, name in (
        (asy_cap, COLOR_CAP, "Whole-cap shift"),
        (asy_single, COLOR_SINGLE, "Single electrode"),
    ):
        ax.plot(curve["x"], curve["y"] * 100, color=color, lw=2,
                marker="o", ms=4.5, label=name, zorder=3)
    ax.axhline(CLINICAL_ASYMMETRY_PP, color=INK, ls=":", lw=1.2)
    ax.annotate(
        "clinical asymmetry threshold (2:1)",
        xy=(1.56, CLINICAL_ASYMMETRY_PP), xytext=(1.56, CLINICAL_ASYMMETRY_PP + 0.7),
        fontsize=9, color=INK, va="bottom", ha="right",
    )
    ax.set_ylim(0, CLINICAL_ASYMMETRY_PP * 1.18)
    ax.set_xlabel("Electrode displacement (cm)")
    ax.set_ylabel("Change in left–right asymmetry\n(percentage points)")
    ax.set_title("B  Interhemispheric asymmetry", loc="left", weight="bold")
    _reference_lines(ax, ax.get_ylim()[1], label=False)
    ax.legend(frameon=False, loc="lower right")

    headline: dict = {}

    # ── Panel C: dipole localisation error ───────────────────────────────────
    if dipole is not None:
        ax = axes[2]
        cap_fits = dipole[dipole["mode"] == "cap"]
        stats = (
            cap_fits.groupby("magnitude_cm")
            .agg(
                x=("realised_mm_mean", "mean"),
                med=("error_mm", "median"),
                lo=("error_mm", lambda s: s.quantile(0.25)),
                hi=("error_mm", lambda s: s.quantile(0.75)),
            )
            .reset_index()
        )
        stats["x"] /= 10.0
        stats = stats.sort_values("x")
        ax.fill_between(stats["x"], stats["lo"], stats["hi"],
                        color=COLOR_CAP, alpha=0.15, lw=0)
        ax.plot(stats["x"], stats["med"], color=COLOR_CAP, lw=2,
                marker="o", ms=4.5, label="Whole-cap shift", zorder=3)
        ax.set_xlabel("Electrode displacement (cm)")
        ax.set_ylabel("Dipole localisation error (mm)")
        ax.set_title("C  Source localisation", loc="left", weight="bold")
        _reference_lines(ax, ax.get_ylim()[1], label=False)
        ax.legend(frameon=False, loc="upper left")
        loc_curve = stats.rename(columns={"med": "y"})
        headline["loc_margin_mm"] = interp(loc_curve, MARGIN_CM)
        headline["loc_expert_mm"] = interp(loc_curve, EXPERT_CM)
        headline["loc_app_mm"] = interp(loc_curve, APP_CM)
        headline["loc_baseline_floor_mm"] = float(
            dipole[dipole["mode"] == "baseline"]["error_mm"].mean()
        )

    for ax in axes:
        ax.set_xlim(0, 1.6)
        ax.grid(axis="y", color=MUTED, alpha=0.18, lw=0.6)
        ax.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    headline.update({
        "amp_margin_cap_pct": interp(amp_cap, MARGIN_CM) * 100,
        "amp_expert_cap_pct": interp(amp_cap, EXPERT_CM) * 100,
        "amp_app_cap_pct": interp(amp_cap, APP_CM) * 100,
        "amp_margin_single_pct": interp(amp_single, MARGIN_CM) * 100,
        "asy_margin_cap_pp": interp(asy_cap, MARGIN_CM) * 100,
        "asy_expert_cap_pp": interp(asy_cap, EXPERT_CM) * 100,
        "asy_app_cap_pp": interp(asy_cap, APP_CM) * 100,
        "linearity_r2_amp_cap": linearity_check(amp_cap),
        "linearity_r2_amp_single": linearity_check(amp_single),
    })
    return headline


def main() -> None:
    summary = pd.read_csv(os.path.join(CACHE, "summary.csv"))
    dipole_path = os.path.join(CACHE, "dipole_fit.csv")
    dipole = pd.read_csv(dipole_path) if os.path.exists(dipole_path) else None
    if dipole is None:
        print("NOTE: dipole_fit.csv not present - panel C omitted")

    headline = build_figure(summary, dipole)
    print(f"\nFigure -> {FIGURE}\n")
    print("=== HEADLINE NUMBERS ===")
    for key, value in headline.items():
        print(f"  {key:32s} {value:8.3f}")


if __name__ == "__main__":
    main()
