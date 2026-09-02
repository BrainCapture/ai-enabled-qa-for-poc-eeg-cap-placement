#!/usr/bin/env python3
"""Supplementary analyses accompanying the clinical paper.

Three analyses, all from data already in hand:

1. Per-electrode mean absolute error by arm.
2. Participant characteristics against positioning accuracy.
3. Description of the two incorrect placements.

Parsing, the subject-exclusion rule and the electrode keys are reused from
`analyze_study` so this cannot drift from the published analysis. `_verify`
re-derives the manuscript's headline numbers and fails loudly if they move.

On analysis 2: the outcome is the *continuous* per-participant MAE, not the
three-level placement rating. With only two Incorrect trials in the study, a
test of demographics against the rating has essentially no power, and a null
result would say nothing about whether hair or head size affects placement.
Every participant contributes information to the continuous outcome.

Run:  python3 revision_analyses.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

import analyze_study as base  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "part2-analyses.md"
FIG = Path(__file__).parent / "outputs" / "19_characteristics_vs_error.png"

#: Rows of the figure, in order, keyed by `Characteristic` in the results
#: table. Only the two a priori hypotheses — head size and hair — are drawn.
#: Hair texture is included because it reaches nominal significance and is
#: discussed in the text below; leaving the one uncomfortable result out of
#: the figure while keeping it in the table would be selective presentation.
FIG_ROWS = {
    "Preauricular arc (cm)": "Head size\n(preauricular arc)",
    "Nasion–inion arc (cm)": "Head size\n(nasion–inion arc)",
    "Hair length": "Hair length\n(medium/long, n = 17)",
    "Hair texture": "Hair texture\n(curly/coily, n = 3)",
}

ABS_COLS = [f"{e}_true_abs" for e in base.ELEC_KEYS]

#: Binary participant characteristics, as simplified by `analyze_study`.
CATEGORICAL = {
    "sex": "Sex",
    "hair_texture_s": "Hair texture",
    "hair_density_s": "Hair density",
    "hair_diameter_s": "Strand diameter",
    "hair_length_s": "Hair length",
    # Reported even though it cannot be tested (28 Loose vs 1 Fixed): showing
    # that it was considered and why it was not tested is more informative than
    # omitting it silently.
    "hair_styling_s": "Structural styling",
}
CONTINUOUS = {
    "age": "Age (years)",
    "preauricular_arc": "Preauricular arc (cm)",
    "nasion_inion_arc": "Nasion–inion arc (cm)",
}

#: A group smaller than this cannot support a rank test worth reporting.
MIN_GROUP = 3


# ── Data ─────────────────────────────────────────────────────────────────────


def _parse_reference_measures() -> pd.DataFrame:
    """Head-size reference arcs (preauricular, nasion-inion) per subject/trial."""
    return base._read_csv("reference_arcs.csv")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trial-level frame and participant metadata, after the published exclusion."""
    meta = base._parse_metadata()
    results = base._parse_results()
    trial_order = base._parse_trial_order()
    reference = _parse_reference_measures()

    df_full = (
        results.merge(trial_order, on=["subject_id", "trial"], how="left")
        .merge(reference, on=["subject_id", "trial"], how="left")
        .merge(meta, on="subject_id", how="left")
    )

    dev_cols = [f"{e}_dev" for e in base.ELEC_KEYS]
    incomplete = df_full.groupby("subject_id", group_keys=False).apply(
        lambda g: g[dev_cols].isna().all(axis=1).any()
    )
    excluded = incomplete[incomplete].index.tolist()

    df = base._add_true_abs(df_full[~df_full["subject_id"].isin(excluded)].copy())
    df["mae"] = df[ABS_COLS].mean(axis=1)
    meta = meta[~meta["subject_id"].isin(excluded)].copy()
    return df, meta


def _verify(df: pd.DataFrame) -> None:
    """Re-derive the manuscript's headline numbers; abort if they have moved."""
    checks = []
    for method, expected in (("Pro", 0.855), ("App", 0.938)):
        got = df[df["method_simple"] == method]["mae"].mean()
        checks.append((f"MAE {method}", got, expected, 0.001))

    pivot = df.pivot_table(index="subject_id", columns="method_simple", values="mae")
    diff = (pivot["App"] - pivot["Pro"]).dropna()
    checks.append(("paired difference", diff.mean(), 0.084, 0.001))

    counts = pd.crosstab(df["method_simple"], df["conclusion"])
    checks.append(("Expert Incorrect", counts.loc["Pro"].get("Forkert", 0), 0, 0.5))
    checks.append(("App Incorrect", counts.loc["App"].get("Forkert", 0), 2, 0.5))
    checks.append(("n trials", len(df), 60, 0.5))

    for name, got, expected, tol in checks:
        if abs(got - expected) > tol:
            raise AssertionError(
                f"{name}: got {got}, manuscript reports {expected}. "
                "The analysis pipeline no longer reproduces the paper."
            )
    print(f"Verified against manuscript: {len(checks)} headline values reproduce.")


# ── Analysis 1: per-electrode error ──────────────────────────────────────────


def per_electrode(df: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute and signed error per electrode, per arm, with paired test."""
    rows = []
    for key in base.ELEC_KEYS:
        record = {"electrode": base.ELECTRODE_LABELS[key]}
        for method, label in (("Pro", "Expert"), ("App", "App-guided")):
            sub = df[df["method_simple"] == method]
            record[f"{label} MAE"] = sub[f"{key}_true_abs"].mean()
            record[f"{label} SD"] = sub[f"{key}_true_abs"].std()
            record[f"{label} signed"] = sub[f"{key}_dev"].mean()

        pivot = df.pivot_table(
            index="subject_id", columns="method_simple", values=f"{key}_true_abs"
        ).dropna()
        record["Δ (App − Expert)"] = (pivot["App"] - pivot["Pro"]).mean()
        record["p (Wilcoxon)"] = stats.wilcoxon(pivot["App"], pivot["Pro"]).pvalue
        rows.append(record)
    return pd.DataFrame(rows)


# ── Analysis 2: participant characteristics ──────────────────────────────────


def _fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        running = min(running, p[idx] * len(p) / rank)
        adjusted[idx] = running
    return adjusted.tolist()


#: Bootstrap replicates for the confidence intervals drawn in the figure. The
#: intervals are illustrative of how little this cohort constrains any of these
#: effects; the reported inference is the rank test and its FDR-adjusted value.
N_BOOT = 5000
BOOT_SEED = 0


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation for `a` against `b`, on Spearman's scale.

    Positive means group `a` tends to have the larger error. Puts the binary
    contrasts on the same [-1, 1] axis as the Spearman coefficients so a single
    figure can carry both.
    """
    u = stats.mannwhitneyu(a, b).statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def _boot_ci(fn, a: np.ndarray, b: np.ndarray, *, paired: bool = True,
             ) -> tuple[float, float]:
    """Percentile bootstrap CI. Paired resamples observation pairs; unpaired
    resamples within each group, preserving group sizes."""
    rng = np.random.default_rng(BOOT_SEED)
    reps = []
    for _ in range(N_BOOT):
        if paired:
            idx = rng.integers(0, len(a), len(a))
            sample = (a[idx], b[idx])
        else:
            sample = (a[rng.integers(0, len(a), len(a))],
                      b[rng.integers(0, len(b), len(b))])
        if min(len(np.unique(sample[0])), len(np.unique(sample[1]))) < 2:
            continue
        reps.append(fn(*sample))
    reps = np.asarray(reps, dtype=float)
    reps = reps[np.isfinite(reps)]
    if len(reps) < N_BOOT // 10:
        return float("nan"), float("nan")
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """Participant characteristics against per-participant MAE, by arm.

    Exploratory. Spearman for continuous predictors, Mann-Whitney for binary
    ones; Benjamini-Hochberg across all tests reported here.
    """
    rows = []
    for method, label in (("Pro", "Expert"), ("App", "App-guided")):
        sub = df[df["method_simple"] == method]

        for col, name in CONTINUOUS.items():
            valid = sub[[col, "mae"]].dropna()
            if len(valid) < MIN_GROUP:
                continue
            rho, p = stats.spearmanr(valid[col], valid["mae"])
            lo, hi = _boot_ci(_spearman_rho, valid[col].to_numpy(),
                              valid["mae"].to_numpy())
            rows.append({
                "Arm": label, "Characteristic": name, "Test": "Spearman",
                "n": len(valid), "Effect": f"rho = {rho:+.2f}", "p": p,
                "effect_size": rho, "ci_lo": lo, "ci_hi": hi,
                "Label": name.split(" (")[0], "n_min": len(valid),
            })

        for col, name in CATEGORICAL.items():
            valid = sub[[col, "mae"]].dropna()
            groups = valid.groupby(col)["mae"]
            if len(groups) != 2 or groups.size().min() < MIN_GROUP:
                rows.append({
                    "Arm": label, "Characteristic": name, "Test": "not tested",
                    "n": len(valid),
                    "Effect": f"group too small (min n = {groups.size().min() if len(groups) else 0})",
                    "p": np.nan, "effect_size": np.nan,
                    "ci_lo": np.nan, "ci_hi": np.nan,
                    "Label": name, "n_min": int(groups.size().min()) if len(groups) else 0,
                })
                continue
            (name_a, a), (name_b, b) = list(groups)
            p = stats.mannwhitneyu(a, b).pvalue
            lo, hi = _boot_ci(_rank_biserial, a.to_numpy(), b.to_numpy(),
                              paired=False)
            rows.append({
                "Arm": label, "Characteristic": name, "Test": "Mann–Whitney",
                "n": len(valid),
                "Effect": f"{name_a} {a.mean():.2f} vs {name_b} {b.mean():.2f} cm",
                "p": p,
                "effect_size": _rank_biserial(a.to_numpy(), b.to_numpy()),
                "ci_lo": lo, "ci_hi": hi,
                "Label": f"{name} ({name_a.lower()}, n = {len(a)})",
                "n_min": int(groups.size().min()),
            })

    out = pd.DataFrame(rows)
    tested = out["p"].notna()
    out.loc[tested, "p (FDR)"] = _fdr(out.loc[tested, "p"].tolist())
    return out


def characteristics_figure(chars: pd.DataFrame) -> Path:
    """Two a priori hypotheses — head size and hair — on one axis.

    Restricted to those two. The full set of characteristics is in the
    table above; a row per characteristic would give eighteen null estimates
    the visual weight of a result. Continuous predictors contribute Spearman's
    rho and binary ones the rank-biserial correlation, which share the [-1, 1]
    scale, so the rows are comparable. Statistics come from `chars`, so the
    figure cannot drift from the table beside it.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
    ypos = {c: len(FIG_ROWS) - 1 - i for i, c in enumerate(FIG_ROWS)}

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.axvline(0, color="0.35", lw=1.0, zorder=1)

    for method, label, offset in (("Pro", "Expert", 0.16), ("App", "App-guided", -0.16)):
        arm = chars[chars["Arm"] == label]
        colour = base.METHOD_SIMPLE_PAL[method]
        drawn = False
        for _, r in arm.iterrows():
            if r["Characteristic"] not in ypos:
                continue
            y = ypos[r["Characteristic"]] + offset
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y],
                    color=colour, lw=1.6, alpha=0.45, solid_capstyle="round", zorder=2)
            ax.scatter(r["effect_size"], y, color=colour, s=58, zorder=3,
                       edgecolors="white", linewidth=1.0,
                       label=label if not drawn else None)
            drawn = True
            if r["p"] < 0.05:
                ax.annotate(f"p = {r['p']:.3f}, FDR {r['p (FDR)']:.2f}",
                            (r["ci_hi"], y), xytext=(7, 0),
                            textcoords="offset points", va="center",
                            fontsize=8.5, color="0.25")

    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([FIG_ROWS[c] for c in ypos])
    ax.set_ylim(-0.5, len(FIG_ROWS) - 0.5)
    ax.set_xlim(-1.05, 1.35)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("← lower error          rank association          higher error →",
                  fontsize=10, color="0.25")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150, facecolor="white")
    plt.close(fig)
    return FIG


# ── Analysis 3: the two incorrect placements ─────────────────────────────────


def failures(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-electrode profile and characteristics of every Incorrect trial."""
    bad = df[df["conclusion"] == "Forkert"]
    profile = pd.DataFrame({
        "Electrode": [base.ELECTRODE_LABELS[k] for k in base.ELEC_KEYS],
        **{
            f"Subject {int(r.subject_id)} ({base.METHOD_DISPLAY.get(r.method, r.method)})":
                [r[f"{k}_dev"] for k in base.ELEC_KEYS]
            for _, r in bad.iterrows()
        },
    })
    cols = ["subject_id", "method", "mae", "age", "sex",
            "hair_texture_s", "hair_density_s", "hair_length_s",
            "preauricular_arc", "nasion_inion_arc"]
    return profile, bad[cols].reset_index(drop=True)


# ── Reporting ────────────────────────────────────────────────────────────────


def _fmt_p(value: float) -> str:
    if pd.isna(value):
        return "—"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    df, _ = load()
    _verify(df)

    elec = per_electrode(df)
    chars = characteristics(df)
    profile, failure_meta = failures(df)
    figure = characteristics_figure(chars)

    cohort_mae = df.groupby("method_simple")["mae"].agg(["mean", "std"])

    lines: list[str] = []
    add = lines.append

    add("# Part 2 analyses — per-electrode errors, participant characteristics,")
    add("# and the two incorrect placements")
    add("")
    add("Generated by `clinical/revision_analyses.py`, which reuses the")
    add("parsing and subject-exclusion rules of `analyze_study.py` and verifies")
    add("that it still reproduces the manuscript's headline numbers before")
    add("computing anything new.")
    add("")
    add(f"n = {df['subject_id'].nunique()} participants, {len(df)} trials.")
    add("")

    add("## 1. Per-electrode error by arm")
    add("")
    add("Mean absolute error (cm) per measured position. *Signed* columns give")
    add("the mean directional deviation (measured − expected); a positive value")
    add("means placement too far from the reference landmark. The Wilcoxon test")
    add("is paired within participant and is exploratory — the study was not")
    add("powered for per-electrode comparisons.")
    add("")
    header = ["Electrode", "Expert MAE", "App MAE", "Δ", "p",
              "Expert signed", "App signed"]
    add("| " + " | ".join(header) + " |")
    add("|" + "---|" * len(header))
    for _, r in elec.iterrows():
        add(
            f"| {r['electrode']} "
            f"| {r['Expert MAE']:.2f} ± {r['Expert SD']:.2f} "
            f"| {r['App-guided MAE']:.2f} ± {r['App-guided SD']:.2f} "
            f"| {r['Δ (App − Expert)']:+.2f} "
            f"| {_fmt_p(r['p (Wilcoxon)'])} "
            f"| {r['Expert signed']:+.2f} "
            f"| {r['App-guided signed']:+.2f} |"
        )
    add("")
    add(f"Pooled across all ten positions: Expert "
        f"{cohort_mae.loc['Pro', 'mean']:.3f} cm, App-guided "
        f"{cohort_mae.loc['App', 'mean']:.3f} cm.")
    add("")

    # How much of the overall arm difference is attributable to T7/T8? The
    # pooled MAE is the mean over the ten positions, so each position's delta
    # contributes exactly one tenth of itself to the total.
    deltas = elec.set_index("electrode")["Δ (App − Expert)"]
    total = deltas.mean()
    temporal = deltas[["T7", "T8"]].sum() / len(deltas)
    add(f"**T7 and T8 account for {temporal / total:.0%} of the overall "
        f"difference between arms** ({temporal:+.3f} cm of {total:+.3f} cm). "
        "Both electrodes are displaced outward in *every one of the 60 trials*, "
        "in both arms (T7 minimum +0.20 cm, T8 minimum +0.90 cm): not one "
        "placement, expert or guided, put either temporal electrode short of "
        "its reference position. A deviation that never changes sign across 60 "
        "independent placements by different operators is the signature of a "
        "fixed geometric offset in the cap, not of variable user error.")
    add("")
    add("Two qualifications the manuscript should carry. First, although the "
        "temporal bias is present in both arms, it is significantly *larger* "
        "under App-guided placement (T7 +0.42 cm, p = 0.003; T8 +0.30 cm, "
        "p = 0.019), so it is not purely a cap-geometry effect independent of "
        "who placed the cap. Second, the arms differ in opposite directions at "
        "the two frontal anteroposterior positions — App-guided is better at "
        "Fp1 and worse at Fp2 — which is difficult to attribute to a systematic "
        "cause and is most likely noise at this sample size.")
    add("")

    add("## 2. Participant characteristics vs. positioning accuracy")
    add("")
    add("Outcome is per-participant mean absolute error across all ten")
    add("positions. Exploratory: p-values are unadjusted, with")
    add("Benjamini–Hochberg values alongside. Head circumference was not")
    add("recorded in the study database, so the preauricular and nasion–inion")
    add("arcs — the measurements from which expected electrode positions were")
    add("derived — are used as head-size covariates.")
    add("")
    header = ["Arm", "Characteristic", "Test", "n", "Effect", "p", "p (FDR)"]
    add("| " + " | ".join(header) + " |")
    add("|" + "---|" * len(header))
    for _, r in chars.iterrows():
        add(
            f"| {r['Arm']} | {r['Characteristic']} | {r['Test']} | {r['n']} "
            f"| {r['Effect']} | {_fmt_p(r['p'])} "
            f"| {_fmt_p(r.get('p (FDR)', np.nan))} |"
        )
    add("")
    add(f"![Participant characteristics vs. positioning error]"
        f"({os.path.relpath(figure, OUT.parent)})")
    add("")
    add("*Figure S1. Two a priori hypotheses — head size and hair —")
    add("against per-participant mean absolute positioning error, by arm.")
    add("Points are rank associations on a common scale (Spearman\'s ρ for the")
    add("arcs, rank-biserial r for the named hair group); positive means higher")
    add("error, and bars are 95% percentile bootstrap intervals. Two")
    add("associations separate from zero, both in the App-guided arm, and")
    add("neither survives Benjamini–Hochberg correction; the curly/coily")
    add("contrast rests on three participants. The remaining characteristics")
    add("are in the table above and are null in both arms.*")
    add("")
    add("**No characteristic survives correction for multiple comparisons.**")
    add("Two associations reach nominal significance before correction, both in")
    add("the App-guided arm and both worth reporting honestly rather than")
    add("suppressing:")
    add("")
    add("- *Head size.* Positioning error rises with preauricular arc under")
    add("  App-guided placement (rho = +0.43, p = 0.017) but not under Expert")
    add("  placement (rho = +0.24, p = 0.21). This is the expected direction for")
    add("  the head-size hypothesis, and it is the one signal pointing that way, but it")
    add("  does not survive correction (FDR p = 0.25) and the arm difference is")
    add("  itself untested.")
    add("- *Hair texture.* Curly/coily hair was associated with **lower** error")
    add("  (0.75 vs 0.96 cm, p = 0.032) — the opposite of the expected")
    add("  direction. This rests on three participants and should not be")
    add("  interpreted; it is reported only to avoid selective presentation.")
    add("")
    add("Hair length, the other a priori hypothesis, shows no association in")
    add("either arm (App-guided p = 0.74). Structural styling could not be")
    add("tested: 28 of 29 participants with the field recorded wore hair loose.")
    add("")
    add("The honest summary for the manuscript is that this cohort was neither")
    add("designed nor powered to detect demographic effects on placement")
    add("accuracy, that no effect survives correction, and that performance in")
    add("participants with dense curly or coily hair remains genuinely")
    add("uncertain on three participants.")
    add("")

    add("## 3. The two incorrect placements")
    add("")
    add("Signed deviation (cm) per position for each trial rated Incorrect.")
    add("")
    add("| " + " | ".join(profile.columns) + " |")
    add("|" + "---|" * len(profile.columns))
    for _, r in profile.iterrows():
        cells = [str(r.iloc[0])] + [f"{v:+.2f}" for v in r.iloc[1:]]
        add("| " + " | ".join(cells) + " |")
    add("")
    add("Characteristics of the two participants:")
    add("")
    add("```")
    add(failure_meta.to_string(index=False))
    add("```")
    add("")
    add("**Both failures occurred in the self-guided sub-condition** (2 of 19,")
    add("11%); there were none among the 11 helper-guided placements and none")
    add("among the 30 expert placements. That is the clearest common factor.")
    add("")
    add("The two profiles are not alike, so a single mechanism is unlikely.")
    add("Subject 19 shows a large, coherent displacement — T7 +4.1 cm with both")
    add("occipital positions displaced laterally by roughly 3 cm — consistent")
    add("with the whole cap sitting rotated and offset on the head. Subject 1")
    add("shows a smaller, more diffuse pattern whose largest components are the")
    add("two occipital anteroposterior positions (both −2.2 cm), consistent with")
    add("the cap sitting too far forward rather than rotated. Both are")
    add("consistent with the ergonomic difficulty of self-placement, where the")
    add("participant cannot see the posterior electrodes while adjusting them.")
    add("")
    add("> The application's own verdict at the time of placement was **not")
    add("> recorded** in the study database, so whether the app had indicated")
    add("> acceptable placement cannot be determined from these data. This is a")
    add("> limitation of the study design: a false-negative of the")
    add("> quality-assurance function is the failure mode that matters most, and")
    add("> it cannot be assessed retrospectively. Prospective logging of the")
    add("> application's own output should be a requirement for future studies.")
    add("")

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
