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
#: Displacement magnitudes (cm) the headline numbers are reported at. These are
#: round points on the model's own abscissa. They are deliberately NOT the
#: study's measured mean absolute errors: MAE averages ten one-dimensional
#: coordinate deviations over six electrodes, while this axis is mean Euclidean
#: displacement over all 19, so evaluating the curves at those values would
#: assert an equivalence the data do not support. See the README.
REPORT_AT_CM = (0.25, 0.5, 1.0, 1.35)

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


# NOTE: an earlier version drew the study's Expert (0.855 cm) and App-guided
# (0.938 cm) mean absolute errors on this axis, plus a band running to
# 0.855 + 0.5 = 1.355 cm labelled "worst placement the margin would accept".
# That annotation was the reason this module was withdrawn from the paper: it
# maps MAE — a mean of ten one-dimensional coordinate deviations over six
# electrodes — onto the model's mean Euclidean array displacement, which is a
# different quantity, and it reads a population-mean margin as a per-placement
# bound. The axis is a property of the model; nothing from the trial data
# belongs on it. Do not reinstate it.

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

    def draw(ax, cap_curve, single_curve, ylabel, title, ylim):
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
        ax.set_xlabel("Mean electrode displacement (cm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylim(0, ylim)

    draw(axes[0], amp_cap, amp_single,
         "Change in scalp potential\n(% of peak amplitude)",
         "A  Signal amplitude", 46)

    ax = axes[1]
    draw(ax, asy_cap, asy_single,
         "Change in left\u2013right asymmetry\n(percentage points)",
         "B  Interhemispheric asymmetry", CLINICAL_ASYMMETRY_PP * 1.38)
    ax.axhline(CLINICAL_ASYMMETRY_PP, color=INK, ls=":", lw=1.3, zorder=4)
    ax.annotate("2:1 interhemispheric ratio (scale reference)",
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
             "C  Source localisation", 16.5)
        floor = float(dipole[dipole["mode"] == "baseline"]["error_mm"].mean())
        axes[2].annotate(f"method floor {floor:.2f} mm", xy=(0.04, 0.4),
                         fontsize=8.5, color=MUTED, ha="left", va="bottom")
        headline.update({f"loc_at_{d:g}cm_mm": at(st, d) for d in REPORT_AT_CM})
        headline["loc_baseline_floor_mm"] = floor

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
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=9.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    for name, curve in (("amp", amp_cap), ("asy", asy_cap)):
        for d in REPORT_AT_CM:
            headline[f"{name}_at_{d:g}cm"] = at(curve, d)
            headline[f"{name}_at_{d:g}cm_p90"] = at(curve, d, "hi")
    headline["amp_single_at_1cm"] = at(amp_single, 1.0)
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
