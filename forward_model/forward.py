"""Compute the leadfield for every perturbed electrode position, in one pass.

The BEM forward solution is expensive to set up (~47 s) but cheap per additional
electrode (~0.05 s), so every distinct electrode position across every variant is
placed into a *single* montage and solved once. Each variant's 19-channel
leadfield is then assembled by row selection.

This is only valid because MNE's EEG leadfield is not average-referenced — the
potentials are absolute, so each row is independent of which other electrodes
share the montage. `_assert_unreferenced` checks that assumption holds rather
than trusting it. The average reference is applied per configuration downstream
in `metrics.py`.
"""

from __future__ import annotations

import os

import mne
import numpy as np
from geometry import (
    ARRAY_1020,
    BEM_SOL,
    SRC_FILE,
    Variant,
    baseline_positions,
    build_variants,
    load_scalp,
)

CACHE = os.path.join(os.path.dirname(__file__), "cache", "leadfield.npz")

# Positions closer than this are treated as identical for deduplication (1 um).
_DEDUP_DECIMALS = 6


def _assert_unreferenced(gain: np.ndarray) -> None:
    """Fail loudly if the leadfield turns out to be average-referenced."""
    col_sum = np.abs(gain.sum(axis=0))
    col_max = np.abs(gain).max(axis=0)
    if np.median(col_sum / np.maximum(col_max, 1e-30)) < 1e-6:
        raise RuntimeError(
            "Leadfield appears average-referenced across the shared montage; "
            "the single-forward trick would mix true and displaced electrodes."
        )


def _registry(
    base: dict[str, np.ndarray], variants: list[Variant]
) -> tuple[dict[tuple, int], np.ndarray]:
    """Map each distinct electrode position to a row index."""
    index: dict[tuple, int] = {}
    positions: list[np.ndarray] = []

    def add(pos: np.ndarray) -> None:
        key = tuple(np.round(pos, _DEDUP_DECIMALS))
        if key not in index:
            index[key] = len(positions)
            positions.append(pos)

    for name in ARRAY_1020:
        add(base[name])
    for variant in variants:
        for name in ARRAY_1020:
            add(variant.ch_pos[name])
    return index, np.array(positions)


def _row_ids(index: dict[tuple, int], ch_pos: dict[str, np.ndarray]) -> np.ndarray:
    return np.array(
        [
            index[tuple(np.round(ch_pos[name], _DEDUP_DECIMALS))]
            for name in ARRAY_1020
        ]
    )


def compute(n_jobs: int = 8, verbose: bool = True) -> dict:
    """Compute (or load) the leadfield over all perturbed electrode positions."""
    if os.path.exists(CACHE):
        if verbose:
            print(f"Loading cached leadfield from {CACHE}")
        with np.load(CACHE, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}

    mne.set_log_level("ERROR")
    scalp = load_scalp()
    base = baseline_positions(scalp, ARRAY_1020)
    variants = build_variants(base, scalp, ARRAY_1020)
    index, positions = _registry(base, variants)

    if verbose:
        print(f"{len(variants)} variants -> {len(positions)} distinct positions")

    names = [f"E{i:04d}" for i in range(len(positions))]
    ref = mne.channels.make_standard_montage("standard_1020").get_positions()
    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(names, positions, strict=True)),
        coord_frame="mri",
        nasion=ref["nasion"],
        lpa=ref["lpa"],
        rpa=ref["rpa"],
    )
    info = mne.create_info(names, 1000.0, "eeg")
    info.set_montage(montage)

    fwd = mne.make_forward_solution(
        info, trans="fsaverage", src=SRC_FILE, bem=BEM_SOL,
        eeg=True, meg=False, n_jobs=n_jobs,
    )
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)
    gain = fwd["sol"]["data"]
    _assert_unreferenced(gain)

    src = fwd["src"]
    source_rr = np.vstack([s["rr"][s["vertno"]] for s in src])
    hemi = np.concatenate(
        [np.full(len(s["vertno"]), i, dtype=np.int8) for i, s in enumerate(src)]
    )

    out = {
        "gain": gain.astype(np.float32),
        "positions": positions,
        "source_rr": source_rr,
        "source_hemi": hemi,
        "baseline_rows": _row_ids(index, base),
        "variant_keys": np.array([v.key for v in variants]),
        "variant_mode": np.array([v.mode for v in variants]),
        "variant_direction": np.array([v.direction for v in variants]),
        "variant_target": np.array([v.target for v in variants]),
        "variant_magnitude": np.array([v.magnitude_cm for v in variants]),
        "variant_rows": np.array([_row_ids(index, v.ch_pos) for v in variants]),
        "realised_mm": np.array(
            [
                [
                    np.linalg.norm(v.ch_pos[n] - base[n]) * 1000.0
                    for n in ARRAY_1020
                ]
                for v in variants
            ]
        ),
        "array": np.array(ARRAY_1020),
    }

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    if verbose:
        print(f"Saved leadfield {gain.shape} -> {CACHE}")
    return out


def validate_against_standard(data: dict, n_jobs: int = 8) -> float:
    """Check the custom MRI-frame montage reproduces `set_montage('standard_1020')`.

    Returns the max relative deviation between the two leadfields; should be ~0.
    """
    mne.set_log_level("ERROR")
    info = mne.create_info(list(ARRAY_1020), 1000.0, "eeg")
    info.set_montage("standard_1020")
    fwd = mne.make_forward_solution(
        info, trans="fsaverage", src=SRC_FILE, bem=BEM_SOL,
        eeg=True, meg=False, n_jobs=n_jobs,
    )
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)
    reference = fwd["sol"]["data"]
    ours = data["gain"][data["baseline_rows"]]
    # Both are defined up to a per-source constant; compare average-referenced.
    reference = reference - reference.mean(axis=0, keepdims=True)
    ours = ours - ours.mean(axis=0, keepdims=True)
    return float(
        np.abs(ours - reference).max() / np.abs(reference).max()
    )


if __name__ == "__main__":
    result = compute()
    print("gain:", result["gain"].shape, "sources:", result["source_rr"].shape)
