"""Quantify how far a displaced electrode array changes the recorded EEG.

Every metric compares a perturbed configuration against the nominal 10-20 array
over a population of cortical sources, after applying the average reference
independently to each configuration.

Sources are restricted to those the scalp array can actually see: deep and
medial-wall dipoles project almost nothing to the scalp, so relative error at
those sources is a division by near-zero and would dominate any summary without
being clinically meaningful. `SOURCE_STRENGTH_PERCENTILE` sets the cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from geometry import HOMOLOGOUS_PAIRS

#: Keep sources whose baseline scalp topography is at least this strong.
SOURCE_STRENGTH_PERCENTILE = 50.0

#: Number of sources used for the dipole-scan localisation metric.
N_LOCALISATION_SOURCES = 800
LOCALISATION_SEED = 0

#: A left/right amplitude ratio of 2 - the usual clinical threshold for calling
#: an interhemispheric asymmetry - corresponds to this asymmetry index.
CLINICAL_ASYMMETRY_INDEX = 1.0 / 3.0


def average_reference(gain: np.ndarray) -> np.ndarray:
    return gain - gain.mean(axis=0, keepdims=True)


def _unit(gain: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(gain, axis=0, keepdims=True)
    return gain / np.maximum(norm, 1e-30)


def strong_sources(g0: np.ndarray) -> np.ndarray:
    """Boolean mask of sources with an above-threshold scalp projection."""
    strength = np.linalg.norm(g0, axis=0)
    return strength >= np.percentile(strength, SOURCE_STRENGTH_PERCENTILE)


def asymmetry_index(gain: np.ndarray, array: list[str]) -> np.ndarray:
    """Per-pair left/right asymmetry index, shape (n_pairs, n_sources).

    (|left| - |right|) / (|left| + |right|), bounded in [-1, 1].
    """
    idx = {name: i for i, name in enumerate(array)}
    out = []
    for left, right in HOMOLOGOUS_PAIRS:
        vl = np.abs(gain[idx[left]])
        vr = np.abs(gain[idx[right]])
        out.append((vl - vr) / np.maximum(vl + vr, 1e-30))
    return np.array(out)


def localisation_error_mm(
    g0: np.ndarray,
    gc: np.ndarray,
    source_rr: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Dipole-scan localisation error induced by the electrode displacement.

    Data are generated at the true source with the *displaced* electrodes, then
    localised by scanning the nominal leadfield - i.e. the analyst is unaware the
    cap moved. Returns the distance (mm) between true and recovered source.
    """
    scan = _unit(g0)  # what the analyst assumes
    observed = _unit(gc[:, test_idx])  # what was actually recorded
    similarity = scan.T @ observed  # (n_sources, n_test)
    recovered = np.abs(similarity).argmax(axis=0)
    return np.linalg.norm(source_rr[test_idx] - source_rr[recovered], axis=1) * 1000.0


def evaluate(data: dict, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Compute all metrics for every variant.

    Returns a per-variant summary table and a dict of per-source distributions
    for the configurations used in the figures.
    """
    gain = data["gain"].astype(np.float64)
    array = [str(x) for x in data["array"]]
    source_rr = data["source_rr"]

    g0 = average_reference(gain[data["baseline_rows"]])
    keep = strong_sources(g0)
    g0_keep = g0[:, keep]
    g0_unit = _unit(g0_keep)
    g0_peak = np.abs(g0_keep).max(axis=0)
    ai0 = asymmetry_index(g0_keep, array)

    rng = np.random.default_rng(LOCALISATION_SEED)
    kept_idx = np.flatnonzero(keep)
    test_idx = rng.choice(kept_idx, size=N_LOCALISATION_SOURCES, replace=False)

    ch_index = {name: i for i, name in enumerate(array)}
    rows = []
    distributions: dict[str, np.ndarray] = {}

    n_variants = len(data["variant_keys"])
    for i in range(n_variants):
        key = str(data["variant_keys"][i])
        gc = average_reference(gain[data["variant_rows"][i]])
        gc_keep = gc[:, keep]

        diff = gc_keep - g0_keep
        peak_rel_err = np.abs(diff).max(axis=0) / g0_peak
        rdm = np.linalg.norm(_unit(gc_keep) - g0_unit, axis=0)
        lnmag = np.log(
            np.maximum(np.linalg.norm(gc_keep, axis=0), 1e-30)
            / np.maximum(np.linalg.norm(g0_keep, axis=0), 1e-30)
        )
        d_ai = np.abs(asymmetry_index(gc_keep, array) - ai0)

        target = str(data["variant_target"][i])
        if target in ch_index:
            j = ch_index[target]
            channel_rel_err = np.abs(gc_keep[j] - g0_keep[j]) / g0_peak
        else:
            channel_rel_err = peak_rel_err

        loc_err = localisation_error_mm(g0, gc, source_rr, test_idx)

        realised = data["realised_mm"][i]
        rows.append(
            {
                "key": key,
                "mode": str(data["variant_mode"][i]),
                "direction": str(data["variant_direction"][i]),
                "target": target,
                "magnitude_cm": float(data["variant_magnitude"][i]),
                "realised_mm_mean": float(realised.mean()),
                "realised_mm_max": float(realised.max()),
                "peak_rel_err_median": float(np.median(peak_rel_err)),
                "peak_rel_err_p90": float(np.percentile(peak_rel_err, 90)),
                "channel_rel_err_median": float(np.median(channel_rel_err)),
                "channel_rel_err_p90": float(np.percentile(channel_rel_err, 90)),
                "rdm_median": float(np.median(rdm)),
                "rdm_p90": float(np.percentile(rdm, 90)),
                "lnmag_median": float(np.median(lnmag)),
                "abs_lnmag_p90": float(np.percentile(np.abs(lnmag), 90)),
                "d_asymmetry_median": float(np.median(d_ai)),
                "d_asymmetry_p90": float(np.percentile(d_ai, 90)),
                "d_asymmetry_max": float(d_ai.max()),
                "loc_err_median_mm": float(np.median(loc_err)),
                "loc_err_p90_mm": float(np.percentile(loc_err, 90)),
                "loc_err_mean_mm": float(loc_err.mean()),
            }
        )
        distributions[key] = np.vstack([peak_rel_err, rdm])

        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_variants} variants")

    summary = pd.DataFrame(rows)
    context = {
        "n_sources_total": int(g0.shape[1]),
        "n_sources_kept": int(keep.sum()),
        "test_idx": test_idx,
        "keep": keep,
        "distributions": distributions,
    }
    return summary, context


#: Longitudinal bipolar ("double banana") chain, for the reference-robustness check.
BIPOLAR_CHAIN = [
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]


def reference_robustness(data: dict) -> pd.DataFrame:
    """Amplitude metric recomputed under average, Cz and bipolar references.

    The average reference is a modelling choice; clinical EEG is commonly read in
    a longitudinal bipolar montage. This checks the conclusion is not an artefact
    of that choice.
    """
    gain = data["gain"].astype(np.float64)
    array = [str(x) for x in data["array"]]
    idx = {name: i for i, name in enumerate(array)}

    schemes = {
        "average": average_reference,
        "Cz": lambda g: g - g[idx["Cz"]],
        "bipolar": lambda g: np.array(
            [g[idx[a]] - g[idx[b]] for a, b in BIPOLAR_CHAIN]
        ),
    }

    raw0 = gain[data["baseline_rows"]]
    keep = strong_sources(average_reference(raw0))
    rows = []
    for name, apply_ref in schemes.items():
        g0 = apply_ref(raw0)[:, keep]
        peak0 = np.abs(g0).max(axis=0)
        for i in range(len(data["variant_keys"])):
            if str(data["variant_mode"][i]) != "cap":
                continue
            gc = apply_ref(gain[data["variant_rows"][i]])[:, keep]
            err = np.abs(gc - g0).max(axis=0) / peak0
            rows.append({
                "reference": name,
                "magnitude_cm": float(data["variant_magnitude"][i]),
                "realised_mm_mean": float(data["realised_mm"][i].mean()),
                "peak_rel_err_median": float(np.median(err)),
            })
    return pd.DataFrame(rows)


def sanity_check(data: dict) -> dict:
    """Zero displacement must produce exactly zero error."""
    gain = data["gain"].astype(np.float64)
    g0 = average_reference(gain[data["baseline_rows"]])
    keep = strong_sources(g0)
    loc = localisation_error_mm(
        g0, g0, data["source_rr"], np.flatnonzero(keep)[:200]
    )
    return {
        "self_localisation_max_mm": float(loc.max()),
        "n_sources_kept": int(keep.sum()),
    }
