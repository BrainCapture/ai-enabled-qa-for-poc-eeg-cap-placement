# Data dictionary

Four CSVs, all keyed on `subject_id` (and `trial` where a measurement is
per-trial). Values are in **centimetres** unless stated otherwise.

Read them with `float_precision="round_trip"` — see the note in the root README.

## `participants.csv` — 31 rows, one per enrolled participant

| Column | Type | Notes |
|---|---|---|
| `subject_id` | int | Pseudonymous study identifier, 1–31. |
| `age` | int | Years, 19–71. |
| `sex` | str | `Male` / `Female`, as recorded at enrolment. |
| `hair_texture` | str | `Straight / Wavy` or `Curly / Coily`, with the rater's full descriptor. |
| `hair_density` | str | `Thin / Average` or `High Density`. |
| `hair_diameter` | str | `Fine / Medium` or `Coarse`. |
| `hair_length` | str | `Short / Shaved` or `Medium / Long`. |
| `hair_styling` | str | `Loose` or `Fixed (Braids/Locs)`. |

Hair fields are missing for one participant. `analyze_study._parse_metadata`
collapses each to a two-level `*_s` grouping; that mapping lives in code, not in
the data, so it stays auditable.

## `trial_order.csv` — 100 rows, the randomisation schedule

| Column | Type | Notes |
|---|---|---|
| `subject_id` | int | Includes IDs allocated but never enrolled — join, don't assume. |
| `trial` | int | 1 or 2. Cross-over order was randomised and balanced. |
| `method` | str | `Pro`, `Self + App`, or `Other + App`. |
| `method_simple` | str | `Pro` (expert) or `App` (guided), collapsing the two guided arms. |

## `measurements.csv` — 62 rows (31 participants × 2 trials)

Positioning error for the six measured electrodes, assessed by a blinded
certified EEG expert against IFCN-derived expected positions.

| Column | Type | Notes |
|---|---|---|
| `subject_id`, `trial` | int | Join keys. |
| `conclusion` | str | Blinded rating: `Korrekt` (optimal), `Suboptimalt` (usable), `Forkert` (incorrect). |
| `<E>_dev` | float | **Signed** deviation, measured − expected. Negative is posterior/medial. |
| `<E>_pct_dev` | float | The same deviation as a fraction of the participant's relevant head arc. |

`<E>` ranges over `T7`, `T8`, `Fp1_lat`, `Fp2_lat`, `Fp1_ap`, `Fp2_ap`,
`O1_lat`, `O2_lat`, `O1_ap`, `O2_ap` — `_lat` is the left–right axis and `_ap`
the anteroposterior axis. Unsigned error (`*_true_abs`) is derived in code as
`abs(*_dev)`, not stored.

One participant has a trial with no measurements recorded; the published
analysis excludes that participant entirely, leaving n = 30.

## `reference_arcs.csv` — 62 rows

Head-size reference measurements used to normalise deviations and as the head-
size covariate in the demographic analysis.

| Column | Type | Notes |
|---|---|---|
| `subject_id`, `trial` | int | Join keys. |
| `preauricular_arc` | float | Preauricular arc, 31.5–38.0 cm. |
| `nasion_inion_arc` | float | Nasion → inion via Cz, 33.0–38.0 cm. |

## What is not here

Participant names, dates, free-text notes and the recording file index are
excluded by construction: the exporter emits only the columns listed above.
