# Clinical validation analysis

Reproduces every number, table and figure in the clinical paper.

```bash
python analyze_study.py      # summary statistics, Analyses 1 and 2, figures 01-16
python revision_analyses.py  # per-electrode MAE, demographics, failure cases
python paper_figures.py      # the 300 dpi composites that appear in the manuscript
```

Outputs land in `outputs/`, overwriting the committed copies. Inputs are the
four CSVs in [`data/`](data/) — see [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md).

Requires `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`. No network access.
The whole thing runs in about ten seconds.

## What each script produces

`analyze_study.py` prints the summary statistics block (participants, conclusion
distribution, per-electrode signed and absolute deviations), then the two
pre-specified analyses — Expert vs App-guided, and the three-way Expert vs
Self-guided vs Helper-guided comparison — including the paired Wilcoxon test on
mean absolute error, the T7/T8-excluded sensitivity analysis, Stuart–Maxwell
tests and linearly weighted Cohen's κ. It writes 16 screen-resolution figures.

`revision_analyses.py` covers supplementary analyses: per-electrode mean
absolute error by arm, participant characteristics against positioning
accuracy, and a description of the two incorrect placements. It re-derives the
manuscript's headline numbers first and fails loudly if they move. Its prose
output is written to `outputs/part2-analyses.md`.

`paper_figures.py` writes the publication composites at 300 dpi —
`figure2_combined.png` (Figure 2) and `supplementary4_threearm.png`
(Supplementary Material 4), plus the Bland–Altman panel on its own, which is
not a manuscript figure but is easier to read separated from the confusion
matrix. Both composites are byte-identical to the files submitted with the
manuscript. Its data loader is deliberately **independent** of
`analyze_study.py`: the manuscript figures and the reported statistics reach
the same numbers by two separate code paths from the same CSVs, so a mistake in
either one shows up as a disagreement rather than as a consistent error.

## Checking a reproduction

`outputs/` is committed, holding exactly the files the published manuscript
used. Rerunning the scripts overwrites them in place, so

```bash
git diff --stat clinical/outputs/
```

is the reproduction check: an empty diff means byte-identical output. Figures
are rendered by matplotlib, so a different matplotlib or font version will
perturb pixels without changing any number — read `part2-analyses.md` and the
console output, not the PNG bytes, when the versions differ.

## A note on the exclusion

One participant has a trial with no measurements recorded. The published
analysis drops that participant entirely rather than the single trial, so the
cross-over design stays balanced; all three scripts apply the same rule, and it
reduces the analysed sample from 31 to 30.
