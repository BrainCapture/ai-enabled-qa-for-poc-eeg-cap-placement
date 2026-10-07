#!/usr/bin/env python3
"""
Measurement-protocol figure — version 2 (simplified, metric-accurate).

Distinguishes the two roles of the measurements:
  • SCORED electrode-placement offsets (bold blue) — the deviations that make up
    the reported error (10 coordinate deviations across 6 electrodes):
        1  T7 / T8 offset            (transverse)          T7, T8
        2  Fp / O vertical offset    (nasion→Fp, inion→O)  Fp1, Fp2, O1, O2
        3  Fp / O horizontal offset  (nasion→Fp, inion→O)  Fp1, Fp2, O1, O2
  • HEAD-SIZE references (grey) — used only to define each electrode's expected
    (10 %) target; NOT part of the error:
        4  Transverse arc (LPA–Cz–RPA)
        5  A–P arc via Cz (nasion–Cz–inion)
        6  A–P arc via LPA / RPA
        7  Head circumference

Self-contained schematic — it draws the protocol, not the data, so it
needs no inputs. Writes supplementary2_measurement_schematic.png at
300 dpi: Supplementary Material 2 of the paper.

The committed PNG is the one submitted to the journal. Re-running this
reproduces it pixel for pixel but not byte for byte, since the PNG stream
depends on the matplotlib and font versions in use.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
import numpy as np

HERE = Path(__file__).parent
OUT  = HERE / "supplementary2_measurement_schematic.png"

# ── Palette ───────────────────────────────────────────────────────────────────
C_HEAD_EDGE = "#3A3F44"
C_HEAD_FILL = "#F7F7F5"
C_ELEC      = "#C0392B"     # measured electrode
C_LAND      = "#8E9296"     # reference landmark
C_SCORED    = "#1F6FB2"     # scored offset (error metric)
C_REF       = "#AAB1B7"     # head-size reference (not scored)
INK         = "#2A2E33"
MUTE        = "#71757A"

R = 0.46
FS_LAND, FS_ELEC, FS_NUM, FS_KEY, FS_HDR = 11, 13, 9.5, 11, 10.5
Z_REF, Z_MARK, Z_LAND, Z_SC, Z_TXT, Z_TOP, Z_LBL = 3, 6, 7, 8, 9, 11, 12

STYLE_RC = {"font.family": "STIXGeneral", "font.size": 12, "mathtext.fontset": "stix"}


def _pt(theta_deg, frac=1.0):
    a = np.deg2rad(theta_deg)
    return (frac * R * np.cos(a), frac * R * np.sin(a))


def _dot(ax, xy, color, r=0.034, z=Z_MARK):
    ax.add_patch(Circle(xy, r, facecolor=color, edgecolor="white",
                        linewidth=1.3, zorder=z))


def _head(ax):
    ax.add_patch(Circle((0, 0), R, facecolor=C_HEAD_FILL,
                        edgecolor=C_HEAD_EDGE, linewidth=1.9, zorder=2))


def _nose(ax, theta_deg):
    a = np.deg2rad(theta_deg)
    perp = a + np.pi / 2
    tip = (R + 0.085) * np.cos(a), (R + 0.085) * np.sin(a)
    b1  = (R * np.cos(a) + 0.06 * np.cos(perp), R * np.sin(a) + 0.06 * np.sin(perp))
    b2  = (R * np.cos(a) - 0.06 * np.cos(perp), R * np.sin(a) - 0.06 * np.sin(perp))
    ax.plot([b1[0], tip[0], b2[0]], [b1[1], tip[1], b2[1]],
            color=C_HEAD_EDGE, lw=1.9, zorder=3, solid_joinstyle="round")


def _ear(ax, theta_deg):
    a = np.deg2rad(theta_deg)
    ax.add_patch(Ellipse((R * np.cos(a), R * np.sin(a)), 0.05, 0.15,
                        angle=np.rad2deg(a), facecolor=C_HEAD_FILL,
                        edgecolor=C_HEAD_EDGE, linewidth=1.5, zorder=1))


def _offset(ax, p0, p1, rad=0.0):
    """Bold blue double-headed arrow = a scored offset."""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="<|-|>", mutation_scale=13,
        connectionstyle=f"arc3,rad={rad}", color=C_SCORED, lw=2.2,
        shrinkA=0, shrinkB=0, zorder=Z_TOP))


def _num(ax, xy, n, fc):
    ax.text(xy[0], xy[1], str(n), fontsize=FS_NUM, fontweight="bold",
            color="white", ha="center", va="center", zorder=Z_LBL,
            bbox=dict(boxstyle="circle,pad=0.26", fc=fc, ec="white", lw=1.1))


# ─────────────────────────────────────────────────────────────────────────────
# Panel (a): Axial
# ─────────────────────────────────────────────────────────────────────────────

def _panel_axial(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    _head(ax)
    _nose(ax, 90)
    _ear(ax, 180)
    _ear(ax, 0)
    # Midline = A–P arc via Cz (#5), solid grey reference
    ax.plot([0, 0], [-R * 0.97, R * 0.97], color=C_REF, lw=1.6, zorder=Z_REF)

    elec = {"Fp1": _pt(107, 0.80), "Fp2": _pt(73, 0.80),
            "O1":  _pt(253, 0.80), "O2":  _pt(287, 0.80),
            "T7":  _pt(180, 0.86), "T8":  _pt(0, 0.86)}

    # Landmarks
    for xy in [(0, 0), _pt(180), _pt(0)]:
        _dot(ax, xy, C_LAND, r=0.026, z=Z_LAND)
    ax.text(0.04, 0.03, "Cz", color=MUTE, fontsize=FS_LAND, style="italic",
            ha="left", va="bottom", zorder=Z_TXT)
    for ang, txt, dx, dy, ha in [(90, "Nasion", 0, 0.12, "center"),
                                  (270, "Inion", 0, -0.11, "center"),
                                  (180, "LPA", -0.09, 0.0, "right"),
                                  (0, "RPA", 0.09, 0.0, "left")]:
        x, y = _pt(ang)
        ax.text(x + dx, y + dy, txt, color=MUTE, fontsize=FS_LAND,
                style="italic", ha=ha, va="center")

    # Electrodes
    for xy in elec.values():
        _dot(ax, xy, C_ELEC)
    lbl = {"Fp1": (-0.06, 0.02, "right"), "Fp2": (0.06, 0.02, "left"),
           "O1": (-0.06, -0.02, "right"), "O2": (0.06, -0.02, "left"),
           "T7": (0.0, -0.07, "center"),  "T8": (0.0, -0.07, "center")}
    for name, (dx, dy, ha) in lbl.items():
        x, y = elec[name]
        ax.text(x + dx, y + dy, name, color=C_ELEC, fontsize=FS_ELEC,
                fontweight="bold", ha=ha, va="center", zorder=Z_TXT)

    # ── References (grey) ─────────────────────────────────────────────────────
    # 7  Head circumference
    ax.add_patch(Circle((0, 0), R + 0.05, fill=False, edgecolor=C_REF,
                        linewidth=1.4, linestyle=(0, (5, 3)), zorder=Z_REF))
    _num(ax, _pt(215, (R + 0.05) / R), 7, C_REF)
    # 4  Transverse arc LPA–Cz–RPA (thin grey line)
    ax.plot([_pt(180)[0], _pt(0)[0]], [0, 0], color=C_REF, lw=1.6, zorder=Z_REF)
    _num(ax, (R * 0.45, 0), 4, C_REF)
    # 5  A–P arc via Cz (nasion–Cz–inion) — the midline, solid grey
    _num(ax, (0, -R * 0.52), 5, C_REF)

    # ── Scored offsets (blue) ─────────────────────────────────────────────────
    # 1  T7 / T8 offset (LPA→T7, RPA→T8): dimension callout above the line
    yo = 0.10
    for sx, tx in [(-R, elec["T7"][0]), (R, elec["T8"][0])]:
        ax.plot([sx, sx], [0.01, yo], color=C_SCORED, lw=0.8, zorder=Z_SC)
        ax.plot([tx, tx], [0.01, yo], color=C_SCORED, lw=0.8, zorder=Z_SC)
        ax.add_patch(FancyArrowPatch((sx, yo), (tx, yo), arrowstyle="<|-|>",
                    mutation_scale=8, color=C_SCORED, lw=1.6,
                    shrinkA=0, shrinkB=0, zorder=Z_SC))
    _num(ax, (_pt(191, 0.98)[0], yo + 0.055), 1, C_SCORED)
    _num(ax, (_pt(-11, 0.98)[0], yo + 0.055), 1, C_SCORED)
    # 3  Fp / O horizontal offset (midline → each electrode)
    for name in ("Fp1", "Fp2", "O1", "O2"):
        x, y = elec[name]
        _offset(ax, (0, y), (x, y))
    _num(ax, (0, elec["Fp1"][1] + 0.05), 3, C_SCORED)
    _num(ax, (0, elec["O1"][1] - 0.05), 3, C_SCORED)

    ax.set_xlim(-R - 0.24, R + 0.24)
    ax.set_ylim(-R - 0.26, R + 0.24)
    ax.text(-R - 0.24, R + 0.22, "(a)  Axial plane", fontsize=15,
            fontweight="bold", ha="left", va="top", color=INK)


# ─────────────────────────────────────────────────────────────────────────────
# Panel (b): Sagittal (left)
# ─────────────────────────────────────────────────────────────────────────────

def _panel_sagittal(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    _head(ax)
    _nose(ax, 180)

    def arc(frac):
        return _pt(180 * (1 - frac))
    fp, o, cz = arc(0.10), arc(0.90), arc(0.50)
    nasion, inion = _pt(180), _pt(0)
    pa = (0.0, -R * 0.12)

    # ── References (grey) ─────────────────────────────────────────────────────
    # 5  A–P arc via Cz (upper semicircle)
    t = np.linspace(180, 0, 220)
    ax.plot(R * np.cos(np.deg2rad(t)), R * np.sin(np.deg2rad(t)),
            color=C_REF, lw=1.8, zorder=Z_REF)
    _num(ax, _pt(138, 1.0), 5, C_REF)
    # 6  A–P arc via LPA (near-straight)
    ax.add_patch(FancyArrowPatch(nasion, inion, arrowstyle="-",
                connectionstyle="arc3,rad=0.12", color=C_REF, lw=1.5,
                zorder=Z_REF))
    _num(ax, (R * 0.30, -R * 0.095), 6, C_REF)

    # Landmarks
    for xy in (nasion, inion, cz, pa):
        _dot(ax, xy, C_LAND, r=0.026, z=Z_LAND)
    ax.text(nasion[0] - 0.12, 0.0, "Nasion", color=MUTE, fontsize=FS_LAND,
            style="italic", ha="right", va="center")
    ax.text(inion[0] + 0.06, 0.0, "Inion", color=MUTE, fontsize=FS_LAND,
            style="italic", ha="left", va="center")
    ax.text(cz[0], cz[1] + 0.055, "Cz", color=MUTE, fontsize=FS_LAND,
            style="italic", ha="center", va="bottom")
    ax.text(pa[0], pa[1] - 0.05, "LPA", color=MUTE, fontsize=FS_LAND,
            style="italic", ha="center", va="top")

    # Electrodes
    for xy, name, (dx, dy, ha) in [(fp, "Fp1", (-0.055, 0.05, "right")),
                                   (o, "O1", (0.055, 0.05, "left"))]:
        _dot(ax, xy, C_ELEC)
        ax.text(xy[0] + dx, xy[1] + dy, name, color=C_ELEC, fontsize=FS_ELEC,
                fontweight="bold", ha=ha, va="center", zorder=Z_TXT)

    # T7 (transverse offset) — projects just above LPA in a lateral view
    t7 = (pa[0], pa[1] + 0.15)
    _dot(ax, t7, C_ELEC)
    ax.text(t7[0] - 0.05, t7[1], "T7", color=C_ELEC, fontsize=FS_ELEC,
            fontweight="bold", ha="right", va="center", zorder=Z_TXT)

    # ── Scored offsets (blue): 1 T7/T8 (LPA→T7), 2 Fp/O vertical ──────────────
    _offset(ax, pa, t7)
    _num(ax, (t7[0] + 0.065, (pa[1] + t7[1]) / 2), 1, C_SCORED)
    _offset(ax, nasion, fp, rad=-0.30)
    _offset(ax, inion, o, rad=0.30)
    _num(ax, _pt(171, 1.14), 2, C_SCORED)
    _num(ax, _pt(9, 1.14), 2, C_SCORED)

    ax.set_xlim(-R - 0.24, R + 0.24)
    ax.set_ylim(-R - 0.26, R + 0.24)
    ax.text(-R - 0.24, R + 0.22, "(b)  Sagittal plane  (left)", fontsize=15,
            fontweight="bold", ha="left", va="top", color=INK)


# ─────────────────────────────────────────────────────────────────────────────
# Key
# ─────────────────────────────────────────────────────────────────────────────

def _disc(ax, x, y, n, col, ms=14):
    ax.plot(x, y, "o", ms=ms, mfc=col, mec="white", mew=1.1, zorder=3)
    ax.text(x, y, str(n), fontsize=FS_NUM, fontweight="bold", color="white",
            ha="center", va="center", zorder=4)


def _key(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    dx, tx = 0.07, 0.17
    s = 0.072
    y = 0.5 + 6.5 * s   # top; block centred on y = 0.5

    # Markers
    ax.plot(dx, y, "o", ms=13, mfc=C_ELEC, mec="white", mew=1.1)
    ax.text(tx, y, "Measured electrode", fontsize=FS_KEY, va="center", color=INK)
    y -= s
    ax.plot(dx, y, "o", ms=13, mfc=C_LAND, mec="white", mew=1.1)
    ax.text(tx, y, "Reference landmark", fontsize=FS_KEY, va="center", color=INK)
    y -= s

    # Group 1 — scored
    ax.text(0.03, y, "Electrode offset  (scored error)", fontsize=FS_HDR,
            style="italic", color=C_SCORED, va="center")
    y -= s
    for n, desc in [(1, "T7 / T8 offset"),
                    (2, "Fp / O vertical offset"),
                    (3, "Fp / O horizontal offset")]:
        _disc(ax, dx, y, n, C_SCORED)
        ax.text(tx, y, desc, fontsize=FS_KEY, va="center", color=INK)
        y -= s

    # Group 2 — reference
    ax.text(0.03, y, "Head-size reference  (sets target)", fontsize=FS_HDR,
            style="italic", color="#6E767C", va="center")
    y -= s
    for n, desc in [(4, "Transverse arc (LPA–Cz–RPA)"),
                    (5, "A–P arc via Cz (nasion–Cz–inion)"),
                    (6, "A–P arc via LPA / RPA"),
                    (7, "Head circumference")]:
        _disc(ax, dx, y, n, C_REF)
        ax.text(tx, y, desc, fontsize=FS_KEY, va="center", color=INK)
        y -= s


# ─────────────────────────────────────────────────────────────────────────────

def main():
    with plt.rc_context(STYLE_RC):
        fig = plt.figure(figsize=(13.0, 5.6), facecolor="white")
        ax_a = fig.add_axes([0.005, 0.04, 0.375, 0.92])
        ax_b = fig.add_axes([0.375, 0.04, 0.375, 0.92])
        ax_k = fig.add_axes([0.775, 0.04, 0.215, 0.92])
        _panel_axial(ax_a)
        _panel_sagittal(ax_b)
        _key(ax_k)
        ax_k.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, fill=False,
                                     edgecolor="#D5D5D5", lw=1.0,
                                     transform=ax_k.transAxes, clip_on=False))
        fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()
    print(f"  Saved {OUT} (300 dpi)")


if __name__ == "__main__":
    main()
