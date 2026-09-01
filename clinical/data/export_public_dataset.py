#!/usr/bin/env python3
"""Export the public, de-identified study dataset — PUBLISHED FOR AUDIT ONLY.

This is the script that produced the four CSVs sitting beside it. It is
reproduced here verbatim, save for the guard in ``_refuse_to_run`` and the
imports moved inside ``export``, so that the de-identification can be read and
checked rather than taken on trust. **It cannot run from this repository**: it
executes inside the study's private analysis package and reads the source
workbooks under ``data_collection/batch_*/``, neither of which is distributed.

Why the workbooks are not published
-----------------------------------

The participant metadata sheets carry a ``file_name`` -> ``Subject ID`` lookup
with participants' real names in columns 9-13, beside the pseudonymised data
the analysis actually reads. Scrubbing those columns and shipping the workbook
would be a denylist, and a workbook can hide data in other sheets, cell
comments, defined names and cached pivot ranges that a scrub will miss.

This exporter is an allowlist instead. It runs the same parsers the published
analysis uses and writes out exactly the columns those parsers consume, as flat
CSV. Nothing that is not named here can reach the public repository.

Original invocation, inside the private repository:

    python3 export_public_dataset.py [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

#: The only participant columns that may be published. Everything else in the
#: metadata workbook — including the name lookup — is excluded by construction.
PARTICIPANT_COLUMNS = [
    "subject_id", "age", "sex",
    "hair_texture", "hair_density", "hair_diameter", "hair_length", "hair_styling",
]


def _refuse_to_run() -> None:
    """Stop before touching anything. See the module docstring."""
    raise SystemExit(
        "export_public_dataset.py cannot run from the public repository.\n"
        "\n"
        "It is published so the export can be audited, not re-executed. It reads\n"
        "the study's source workbooks under data_collection/batch_*/, which carry\n"
        "a participant name lookup and are deliberately not distributed. The CSVs\n"
        "in this directory are its output; regenerating them is neither possible\n"
        "nor necessary here.\n"
        "\n"
        "Researchers who need the source data should contact the corresponding\n"
        "author — see the data availability statement in the paper."
    )


def export(out_dir: Path) -> dict[str, Path]:
    # Imported lazily, and deliberately: these are the *private* parsers, from
    # the study's own analysis package. They are not the CSV-reading modules of
    # the same name in this repository, and they are not importable here.
    sys.path.insert(0, str(Path(__file__).parent))
    import analyze_study as base
    import revision_analyses as rev

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # Participants — raw allowlisted columns only. The derived `*_s` groupings
    # are recomputed in the public analysis, so the transformation stays visible
    # in code rather than being baked into the data.
    meta = base._parse_metadata()[PARTICIPANT_COLUMNS]
    written["participants"] = _write(meta, out_dir / "participants.csv")

    # Trial order, per-electrode deviations and head-size arcs are all already
    # tidy once parsed; the workbook layout they come from (merged cells, two
    # side-by-side trial blocks, Danish labels) is not worth reproducing.
    written["trial_order"] = _write(base._parse_trial_order(), out_dir / "trial_order.csv")
    written["measurements"] = _write(base._parse_results(), out_dir / "measurements.csv")
    written["reference_arcs"] = _write(rev._parse_reference_measures(),
                                       out_dir / "reference_arcs.csv")
    return written


def _write(df: pd.DataFrame, path: Path) -> Path:
    # Default formatting already writes an exact float64 repr. The matching
    # read side must pass float_precision="round_trip" (see the public
    # analysis's _read_csv) — pandas' default CSV float parser is fast but not
    # correctly rounded, and the last-bit error it introduces is enough to flip
    # a tie in the Wilcoxon sensitivity analysis.
    df.to_csv(path, index=False)
    return path


def main() -> None:
    _refuse_to_run()

    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).parent))
    a = p.parse_args()

    written = export(Path(a.out))
    print(f"Exported to {a.out}\n")
    for name, path in written.items():
        df = pd.read_csv(path)
        print(f"  {path.name:<22} {df.shape[0]:>4} rows x {df.shape[1]:>2} cols")

    # Fail loudly if any published column name looks like a direct identifier.
    banned = ("name", "file", "cpr", "navn", "email", "initial")
    for name, path in written.items():
        cols = [c.lower() for c in pd.read_csv(path, nrows=0).columns]
        bad = [c for c in cols if any(b in c for b in banned)]
        if bad:
            raise SystemExit(f"REFUSING: {path.name} has identifier-like columns {bad}")
    print("\nIdentifier column check: clean.")


if __name__ == "__main__":
    main()
