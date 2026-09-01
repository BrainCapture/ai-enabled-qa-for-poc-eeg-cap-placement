"""Head geometry and electrode-displacement construction for the forward study.

All geometry is handled in the fsaverage MRI (surface RAS) frame, in metres —
the frame that `mne.channels.make_standard_montage("standard_1020")` reports and
that the BEM surfaces in `fsaverage/bem/` live in. Displaced electrodes are built
in that frame and handed to MNE as a `DigMontage`, so the standard
`trans="fsaverage"` head<->MRI transform stays valid.

Displacements are applied in the local tangent plane of the scalp and then
projected back onto the outer-skin surface. Over a head of radius ~9 cm the
chord/arc discrepancy at the largest displacement used here (2 cm) is <0.05 mm,
so the tangential step length is taken as the realised surface displacement;
`realised_displacement_mm` verifies this per variant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mne
import numpy as np
from mne.surface import _project_onto_surface, complete_surface_info

# ── Configuration ────────────────────────────────────────────────────────────

FS_DIR = os.path.join(
    os.path.expanduser("~"), "mne_data", "MNE-fsaverage-data", "fsaverage"
)
BEM_SURF = os.path.join(FS_DIR, "bem", "fsaverage-5120-5120-5120-bem.fif")
BEM_SOL = os.path.join(FS_DIR, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
SRC_FILE = os.path.join(FS_DIR, "bem", "fsaverage-ico-5-src.fif")

FIFFV_BEM_SURF_ID_HEAD = 4

#: Standard international 10-20 clinical array.
ARRAY_1020 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]

#: Homologous left/right pairs, used for the interhemispheric-asymmetry metric.
HOMOLOGOUS_PAIRS = [
    ("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"),
    ("T7", "T8"), ("C3", "C4"), ("P7", "P8"), ("P3", "P4"), ("O1", "O2"),
]

#: Electrodes measured in the clinical study (Supplementary Material 2).
MEASURED = ["Fp1", "Fp2", "T7", "T8", "O1", "O2"]

#: Displacement magnitudes (cm). 0.5 is the non-inferiority margin; 0.855 and
#: 0.938 are the Expert and App-guided mean absolute errors measured in the study.
MAGNITUDES_CM = [0.25, 0.5, 0.855, 0.938, 1.5, 2.0]

#: Anatomical axes in MRI surface RAS.
AXES = {
    "right": np.array([1.0, 0.0, 0.0]),
    "left": np.array([-1.0, 0.0, 0.0]),
    "anterior": np.array([0.0, 1.0, 0.0]),
    "posterior": np.array([0.0, -1.0, 0.0]),
}

#: Rigid whole-cap shift directions. These are the axes the study measured along
#: (anteroposterior and left-right).
CAP_DIRECTIONS = ["anterior", "posterior", "left", "right"]

SUPERIOR = np.array([0.0, 0.0, 1.0])

#: Tangential directions for the single-electrode mode.
N_SINGLE_DIRECTIONS = 8


# ── Scalp surface ────────────────────────────────────────────────────────────


def load_scalp() -> dict:
    """Outer-skin BEM surface, with vertex normals, in MRI frame (metres)."""
    surfs = mne.read_bem_surfaces(BEM_SURF, verbose=False)
    scalp = next(s for s in surfs if s["id"] == FIFFV_BEM_SURF_ID_HEAD)
    return complete_surface_info(scalp, verbose=False)


def project_to_scalp(points: np.ndarray, scalp: dict) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto the scalp surface; return (positions, outward normals)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    _, _, proj, nn = _project_onto_surface(
        points, scalp, project_rrs=True, return_nn=True
    )
    return proj, nn


def fit_sphere(scalp: dict) -> tuple[np.ndarray, float]:
    """Least-squares sphere fit to the scalp vertices; returns (centre, radius) in m."""
    pts = scalp["rr"]
    a = np.hstack([2 * pts, np.ones((len(pts), 1))])
    b = (pts**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = sol[:3]
    radius = float(np.sqrt(sol[3] + centre @ centre))
    return centre, radius


def rotate_about_axis(
    points: np.ndarray, centre: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """Rodrigues rotation of `points` about `axis` through `centre`."""
    k = axis / np.linalg.norm(axis)
    p = np.atleast_2d(points) - centre
    rotated = (
        p * np.cos(angle)
        + np.cross(k, p) * np.sin(angle)
        + k * (p @ k)[:, None] * (1 - np.cos(angle))
    )
    return rotated + centre


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal tangent basis (u, v) for a scalp normal.

    `u` is the anterior direction projected into the tangent plane, so direction
    angle 0 is anterior-ish everywhere on the head and the 8 sampled directions
    are anatomically comparable across electrodes. Where the normal is nearly
    anterior (e.g. Fpz) the superior axis is used instead.
    """
    normal = normal / np.linalg.norm(normal)
    ref = AXES["anterior"]
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    u = ref - np.dot(ref, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


# ── Electrode variants ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Variant:
    """One perturbed array configuration.

    `ch_pos` maps the *base* electrode name to its perturbed position, so every
    variant is directly comparable to the baseline array channel-by-channel.
    """

    key: str
    mode: str  # "cap" | "single"
    direction: str
    magnitude_cm: float
    target: str  # electrode displaced ("all" for a rigid cap shift)
    ch_pos: dict[str, np.ndarray]


def baseline_positions(scalp: dict, array: list[str]) -> dict[str, np.ndarray]:
    """Standard 10-20 positions, projected onto the scalp surface."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_pos = montage.get_positions()["ch_pos"]
    pts = np.array([ch_pos[name] for name in array])
    proj, _ = project_to_scalp(pts, scalp)
    return {name: proj[i] for i, name in enumerate(array)}


def build_variants(
    base: dict[str, np.ndarray],
    scalp: dict,
    array: list[str],
    magnitudes_cm: list[float] = MAGNITUDES_CM,
) -> list[Variant]:
    """Rigid whole-cap shifts and single-electrode displacements."""
    variants: list[Variant] = []

    # Mode 1: rigid cap shift. The manuscript notes that because inter-electrode
    # geometry within the cap is fixed, the dominant error mode is displacement
    # of the whole cap relative to the head. A cap slipping on the head is a
    # rigid *rotation* about the head centre, not a translation: translating the
    # array would drive the frontal electrodes into the scalp and, after
    # re-projection, distort the inter-electrode geometry the cap actually fixes.
    # The rotation angle is set so that arc length at the vertex equals the
    # nominal magnitude; electrodes near the rotation axis move less, which is
    # the physically correct behaviour (see `realised_displacement_mm`).
    centre, radius = fit_sphere(scalp)
    pts = np.array([base[name] for name in array])
    for direction in CAP_DIRECTIONS:
        # Rotating about k = up x shift_direction slides the vertex toward the
        # requested anatomical direction.
        axis = np.cross(SUPERIOR, AXES[direction])
        for mag in magnitudes_cm:
            angle = (mag / 100.0) / radius
            proj, _ = project_to_scalp(
                rotate_about_axis(pts, centre, axis, angle), scalp
            )
            variants.append(
                Variant(
                    key=f"cap|{direction}|{mag}",
                    mode="cap",
                    direction=direction,
                    magnitude_cm=mag,
                    target="all",
                    ch_pos={name: proj[i] for i, name in enumerate(array)},
                )
            )

    # Mode 2: single-electrode displacement in 8 tangential directions —
    # the literal form of the reviewer's question, comparable to Wang & Gotman.
    angles = np.linspace(0, 2 * np.pi, N_SINGLE_DIRECTIONS, endpoint=False)
    for target in MEASURED:
        pos = base[target]
        _, normal = project_to_scalp(pos[None, :], scalp)
        u, v = tangent_basis(normal[0])
        for ai, angle in enumerate(angles):
            step = np.cos(angle) * u + np.sin(angle) * v
            for mag in magnitudes_cm:
                moved, _ = project_to_scalp(
                    (pos + step * (mag / 100.0))[None, :], scalp
                )
                ch_pos = dict(base)
                ch_pos[target] = moved[0]
                variants.append(
                    Variant(
                        key=f"single|{target}|{ai * 360 // N_SINGLE_DIRECTIONS}|{mag}",
                        mode="single",
                        direction=f"{ai * 360 // N_SINGLE_DIRECTIONS}deg",
                        magnitude_cm=mag,
                        target=target,
                        ch_pos=ch_pos,
                    )
                )
    return variants


def realised_displacement_mm(
    base: dict[str, np.ndarray], variant: Variant
) -> np.ndarray:
    """Euclidean distance (mm) between baseline and perturbed electrode positions."""
    return np.array(
        [
            np.linalg.norm(variant.ch_pos[name] - base[name]) * 1000.0
            for name in variant.ch_pos
        ]
    )
