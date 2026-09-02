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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "cache")
FIGURE = os.path.join(HERE, "figures", "forward_displacement.png")

# Matches the palette used by the existing clinical-paper figures (Tol vibrant).
COLOR_CAP = "#0077BB"
COLOR_SINGLE = "#EE7733"
INK = "#222222"
MUTED = "#767676"

#: The margin is prespecified on the paired *difference* in mean absolute
#: positioning error, Delta = MAE(App) - MAE(Expert), not on an absolute
#: displacement. So the placement the margin would still accept as non-inferior
#: is the Expert's own MAE plus the margin, and the margin itself is a distance
#: *along* the x-axis rather than a point on it. The figure is drawn that way.
MARGIN_CM = 0.5
EXPERT_CM = 0.855
APP_CM = 0.938
LIMIT_CM = EXPERT_CM + MARGIN_CM

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
           x_col: str = "realised_mm_mean", scale: float = 100.0):
    """Mean across directions/targets at each displacement, sorted by x.

    `x_col` differs by mode. A cap shift moves every electrode, so the array
    mean is the right abscissa; a single-electrode displacement moves exactly
    one, so the array mean would divide the true displacement by the number of
    channels - use the max (that electrode's own displacement) instead.
    `scale` converts the stored fraction to the plotted unit.
    """
    agg = {"realised": (x_col, "mean"), "y": (value, "mean")}
    if spread:
        agg["hi"] = (spread, "mean")
    out = df.groupby("magnitude_cm").agg(**agg).reset_index()
    out["x"] = out["realised"] / 10.0  # mm -> cm
    out = out.sort_values("x")
    out.attrs["scale"] = scale
    return out


def interp(curve: pd.DataFrame, x: float, col: str = "y") -> float:
    return float(np.interp(x, curve["x"], curve[col]))


def at(curve: pd.DataFrame, x: float, col: str = "y") -> float:
    """Curve value at `x` in plotted units."""
    return interp(curve, x, col) * curve.attrs["scale"]


def linearity_check(curve: pd.DataFrame) -> float:
    """R^2 of a through-origin linear fit; ~1.0 justifies interpolation."""
    x, y = curve["x"].to_numpy(), curve["y"].to_numpy()
    slope = (x @ y) / (x @ x)
    resid = y - slope * x
    return float(1 - resid.var() / y.var())


def _margin_band(ax, label: bool) -> None:
    """Draw the margin as what it is: a span on the x-axis.

    Three marks, three different kinds of claim. Expert MAE is the reference
    standard the study measures against; Expert + margin is the worst placement
    the non-inferiority test would still accept; App-guided is what was actually
    observed. The margin is the gap between the first two.
    """
    ymax = ax.get_ylim()[1]
    ax.axvspan(EXPERT_CM, LIMIT_CM, color=INK, alpha=0.05, lw=0, zorder=0)
    ax.axvline(EXPERT_CM, color=MUTED, ls="--", lw=1.2, zorder=1)
    ax.axvline(LIMIT_CM, color=INK, ls="--", lw=1.2, zorder=1)
    ax.axvline(APP_CM, color=COLOR_CAP, ls="-", lw=1.0, alpha=0.55, zorder=1)

    y = ymax * 0.90
    ax.annotate("", xy=(LIMIT_CM, y), xytext=(EXPERT_CM, y), zorder=6,
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1,
                                shrinkA=0, shrinkB=0))
    ax.annotate("0.5 cm margin", xy=((EXPERT_CM + LIMIT_CM) / 2, y),
                xytext=(0, 5), textcoords="offset points", ha="center",
                va="bottom", fontsize=9, weight="bold", color=INK, zorder=6)
    if label:
        ax.annotate("Expert\n0.855 cm", xy=(EXPERT_CM - 0.04, ymax * 0.99),
                    ha="right", va="top", fontsize=8.5, color=MUTED,
                    linespacing=1.25)
        ax.annotate("worst placement the\nmargin would accept\n1.355 cm",
                    xy=(LIMIT_CM + 0.04, ymax * 0.99), ha="left", va="top",
                    fontsize=8.5, color=INK, linespacing=1.25)


def _readouts(ax, curve, fmt: str, label_app: bool) -> None:
    """Values on the cap curve at the three marked placements."""
    pts = [(EXPERT_CM, MUTED, "o", 6.5, "normal"),
           (APP_CM, COLOR_CAP, "D", 5.5, "normal"),
           (LIMIT_CM, INK, "o", 7.0, "bold")]
    for x, color, marker, size, weight in pts:
        y = at(curve, x)
        ax.plot([x], [y], marker=marker, ms=size, mfc=color, mec="white",
                mew=1.3, zorder=6)
        if marker == "D" and not label_app:
            continue
        # Expert and App-guided sit only 0.083 cm apart, so their labels are
        # pushed to opposite sides of the curve rather than offset along it.
        # The curves rise monotonically, so above-left and below-right are the
        # only sides that stay clear of the line whatever the panel's scale:
        # left of a point the curve is below it, right of it the curve is
        # above. Every offset below must keep to those two quadrants - an
        # above-right label is crossed by the line at slopes as gentle as
        # panel A's, and buried by it at panel C's.
        offsets = {EXPERT_CM: (-11, 6, "right", "bottom"),
                   APP_CM: (6, -9, "left", "top"),
                   LIMIT_CM: (9, -8, "left", "top")}
        dx, dy, ha, va = offsets[x]
        ax.annotate(fmt.format(y), xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=9.5, weight=weight, color=color, zorder=7)


def build_figure(summary: pd.DataFrame, dipole: pd.DataFrame | None) -> dict:
    cap = summary[summary["mode"] == "cap"]
    single = summary[summary["mode"] == "single"]

    amp_cap = _curve(cap, "peak_rel_err_median", "peak_rel_err_p90")
    amp_single = _curve(single, "peak_rel_err_median", x_col="realised_mm_max")
    asy_cap = _curve(cap, "d_asymmetry_median", "d_asymmetry_p90")
    asy_single = _curve(single, "d_asymmetry_median", x_col="realised_mm_max")

    plt.rcParams.update(STYLE)
    n_panels = 3 if dipole is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.4 * n_panels, 4.0))

    def draw(ax, cap_curve, single_curve, ylabel, title, ylim, fmt, first):
        ax.fill_between(cap_curve["x"],
                        cap_curve["y"] * cap_curve.attrs["scale"],
                        cap_curve["hi"] * cap_curve.attrs["scale"],
                        color=COLOR_CAP, alpha=0.15, lw=0, zorder=2)
        ax.plot(cap_curve["x"], cap_curve["y"] * cap_curve.attrs["scale"],
                color=COLOR_CAP, lw=2.2, zorder=3, label="Whole-cap shift")
        if single_curve is not None:
            ax.plot(single_curve["x"],
                    single_curve["y"] * single_curve.attrs["scale"],
                    color=COLOR_SINGLE, lw=1.4, ls=(0, (5, 2)), zorder=3,
                    label="Single electrode")
        ax.set_xlabel("Mean absolute positioning error (cm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylim(0, ylim)
        _margin_band(ax, label=first)
        _readouts(ax, cap_curve, fmt, label_app=first)

    draw(axes[0], amp_cap, amp_single,
         "Change in scalp potential\n(% of peak amplitude)",
         "A  Signal amplitude", 46, "{:.1f}%", True)

    ax = axes[1]
    draw(ax, asy_cap, asy_single,
         "Change in left\u2013right asymmetry\n(percentage points)",
         "B  Interhemispheric asymmetry", CLINICAL_ASYMMETRY_PP * 1.38,
         "{:.1f} pp", False)
    ax.axhline(CLINICAL_ASYMMETRY_PP, color=INK, ls=":", lw=1.3, zorder=4)
    ax.annotate("clinical asymmetry threshold (2:1)",
                xy=(0.04, CLINICAL_ASYMMETRY_PP + 0.6), ha="left", va="bottom",
                fontsize=9, color=INK, zorder=6)

    headline: dict = {}
    if dipole is not None:
        st = (dipole[dipole["mode"] == "cap"].groupby("magnitude_cm")
              .agg(realised=("realised_mm_mean", "mean"),
                   y=("error_mm", "median"),
                   hi=("error_mm", lambda s: s.quantile(0.90)))
              .reset_index())
        st["x"] = st["realised"] / 10.0
        st = st.sort_values("x")
        st.attrs["scale"] = 1.0
        draw(axes[2], st, None, "Dipole localisation error (mm)",
             "C  Source localisation", 16.5, "{:.1f} mm", False)
        floor = float(dipole[dipole["mode"] == "baseline"]["error_mm"].mean())
        axes[2].annotate(f"method floor {floor:.2f} mm", xy=(0.04, 0.4),
                         fontsize=8.5, color=MUTED, ha="left", va="bottom")
        headline.update({
            "loc_expert_mm": at(st, EXPERT_CM),
            "loc_app_mm": at(st, APP_CM),
            "loc_limit_mm": at(st, LIMIT_CM),
            "loc_increment_mm": at(st, LIMIT_CM) - at(st, EXPERT_CM),
            "loc_baseline_floor_mm": floor,
        })

    for ax in axes:
        ax.set_xlim(0, 1.62)
        ax.grid(axis="y", color=MUTED, alpha=0.18, lw=0.6)
        ax.set_axisbelow(True)

    handles = [
        Line2D([], [], color=COLOR_CAP, lw=2.2, label="Whole-cap shift"),
        Patch(facecolor=COLOR_CAP, alpha=0.20, lw=0,
              label="median to 90th percentile across sources"),
        Line2D([], [], color=COLOR_SINGLE, lw=1.4, ls=(0, (5, 2)),
               label="Single electrode (median)"),
        Line2D([], [], color=COLOR_CAP, marker="D", ls="-", lw=1.0, ms=5.5,
               alpha=0.75, label="App-guided, observed (0.938 cm)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=9.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    for name, curve, unit in (("amp", amp_cap, 100), ("asy", asy_cap, 100)):
        headline[f"{name}_expert"] = at(curve, EXPERT_CM)
        headline[f"{name}_app"] = at(curve, APP_CM)
        headline[f"{name}_limit"] = at(curve, LIMIT_CM)
        headline[f"{name}_increment"] = at(curve, LIMIT_CM) - at(curve, EXPERT_CM)
        headline[f"{name}_expert_p90"] = at(curve, EXPERT_CM, "hi")
        headline[f"{name}_limit_p90"] = at(curve, LIMIT_CM, "hi")
    headline["amp_single_at_2cm"] = at(amp_single, 2.0)
    headline["linearity_r2_amp_cap"] = linearity_check(amp_cap)
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
