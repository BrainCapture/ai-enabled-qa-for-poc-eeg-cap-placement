"""Continuous dipole-fit localisation error induced by electrode displacement.

The grid scan in `metrics.py` is limited by the ico-5 source spacing (~3.1 mm):
displacements smaller than half a grid step recover the *same* vertex and report
exactly 0 mm, which understates the error. This module instead runs MNE's
continuous non-linear dipole fit, whose floor is ~0.1 mm (measured by the
"baseline" rows this module emits), so small displacements are resolved.

The scenario modelled: a focal source generates scalp potentials that are
recorded through a *displaced* array, and the analyst then localises the source
assuming the array is where it should be. The resulting mismatch is the
localisation error attributable to cap misplacement.

Fits are batched - `fit_dipole` fits each time sample independently, so many
sources share one setup - and the whole sweep is cached.
"""

from __future__ import annotations

import os

import forward
import geometry as g
import metrics
import mne
import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "cache", "dipole_fit.csv")

#: Cortical sites sampled for the fit, spread over the cortex by k-means.
N_SITES = 12
SITE_SEED = 0

#: Cap-shift directions carried through the dipole fit (the expensive metric).
DIRECTIONS = ["anterior", "left"]


def _nominal_info(data: dict) -> tuple[mne.Info, list[str]]:
    array = [str(x) for x in data["array"]]
    ref = mne.channels.make_standard_montage("standard_1020").get_positions()
    pos = {
        name: data["positions"][data["baseline_rows"][i]]
        for i, name in enumerate(array)
    }
    montage = mne.channels.make_dig_montage(
        ch_pos=pos, coord_frame="mri",
        nasion=ref["nasion"], lpa=ref["lpa"], rpa=ref["rpa"],
    )
    info = mne.create_info(array, 1000.0, "eeg")
    info.set_montage(montage)
    return info, array


def select_sites(data: dict, n_sites: int = N_SITES) -> np.ndarray:
    """Source indices spread over the cortex, restricted to well-seen sources."""
    from scipy.cluster.vq import kmeans2

    gain = data["gain"].astype(np.float64)
    g0 = metrics.average_reference(gain[data["baseline_rows"]])
    keep = np.flatnonzero(metrics.strong_sources(g0))
    rr = data["source_rr"][keep]
    centroids, _ = kmeans2(rr, n_sites, seed=SITE_SEED, minit="++")
    # Nearest actual source to each centroid.
    return np.array(
        [keep[np.linalg.norm(rr - c, axis=1).argmin()] for c in centroids]
    )


def _fit(
    potentials: np.ndarray,
    info: mne.Info,
    cov: mne.Covariance,
    bem,
    true_rr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one dipole per column of `potentials`; return (error_mm, gof)."""
    evoked = mne.EvokedArray(potentials * 1e-9, info.copy(), tmin=0, verbose=False)
    evoked.set_eeg_reference("average", projection=True, verbose=False)
    evoked.apply_proj(verbose=False)
    dip, _ = mne.fit_dipole(evoked, cov, bem, trans="fsaverage", verbose=False)
    # Dipole positions and the forward's source space are both in head coords.
    return np.linalg.norm(dip.pos - true_rr, axis=1) * 1000.0, dip.gof


def run(verbose: bool = True) -> pd.DataFrame:
    if os.path.exists(CACHE):
        if verbose:
            print(f"Loading cached dipole fits from {CACHE}")
        return pd.read_csv(CACHE)

    mne.set_log_level("ERROR")
    data = forward.compute(verbose=False)
    gain = data["gain"].astype(np.float64)
    info, _ = _nominal_info(data)
    cov = mne.make_ad_hoc_cov(info, std=1e-6, verbose=False)
    bem = mne.read_bem_solution(g.BEM_SOL, verbose=False)

    sites = select_sites(data)
    true_rr = data["source_rr"][sites]
    g0 = metrics.average_reference(gain[data["baseline_rows"]])

    rows = []

    # Method floor: fit the undisplaced data with the nominal array.
    err, gof = _fit(g0[:, sites], info, cov, bem, true_rr)
    for i, s in enumerate(sites):
        rows.append({
            "mode": "baseline", "direction": "none", "magnitude_cm": 0.0,
            "site": int(s), "error_mm": float(err[i]), "gof": float(gof[i]),
            "realised_mm_mean": 0.0,
        })
    if verbose:
        print(f"baseline floor: mean {err.mean():.3f} mm, max {err.max():.3f} mm")

    keys = [str(k) for k in data["variant_keys"]]
    for i, key in enumerate(keys):
        if str(data["variant_mode"][i]) != "cap":
            continue
        if str(data["variant_direction"][i]) not in DIRECTIONS:
            continue
        gc = metrics.average_reference(gain[data["variant_rows"][i]])
        err, gof = _fit(gc[:, sites], info, cov, bem, true_rr)
        realised = float(data["realised_mm"][i].mean())
        for j, s in enumerate(sites):
            rows.append({
                "mode": "cap",
                "direction": str(data["variant_direction"][i]),
                "magnitude_cm": float(data["variant_magnitude"][i]),
                "site": int(s), "error_mm": float(err[j]), "gof": float(gof[j]),
                "realised_mm_mean": realised,
            })
        if verbose:
            print(f"  {key}: mean {err.mean():5.2f} mm  median {np.median(err):5.2f} mm")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    out.to_csv(CACHE, index=False)
    return out


if __name__ == "__main__":
    df = run()
    print(df.groupby(["direction", "magnitude_cm"])["error_mm"].describe())
