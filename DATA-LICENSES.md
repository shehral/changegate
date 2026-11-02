# Data licenses

Every dataset changegate loads, downloads or ships a subset of is listed here **before** the
code that uses it is merged. Only permissively licensed data is used. Where a license is
unclear, the battery uses synthetic items instead, marked `source: synthetic`.

A battery item's `source` field must match a `Key` in this table or be `synthetic`.

| Key | Dataset | Revision | Source | License | Subset used | Used by | Recorded |
|---|---|---|---|---|---|---|---|
| `advbench` | AdvBench harmful behaviors (Zou et al., arXiv:2307.15043) | llm-attacks/llm-attacks@098262ed | github.com/llm-attacks/llm-attacks, `data/advbench/harmful_behaviors.csv` | MIT (GitHub license API; the LICENSE text at that commit is copied to `data/prompts/advbench_harmful.LICENSE`) | 128 of 520 `goal` strings, seed 0: `data/prompts/advbench_harmful.jsonl` | `drift.refusal` (harmful set) | 2026-09-27 |
| `synthetic` | Templated harmless instructions | this repository | `scripts/fetch_prompts.py synthetic-harmless` | Apache-2.0 | 128 of 168, seed 0: `data/prompts/synthetic_harmless.jsonl` | `drift.refusal` (harmless set) | 2026-09-27 |
| `synthetic` | Example sycophancy persona spec | this repository | `data/personas/sycophancy.json` | Apache-2.0 | all: 4 + 4 system prompts, 16 questions | `drift.persona` (example only) | 2026-09-27 |
| `strongreject` | StrongREJECT (Souly et al., arXiv:2402.10260) | alexandrasouly/strongreject@f7cad6c1 | github.com/alexandrasouly/strongreject, `strongreject_dataset/strongreject_dataset.csv` (sha256 4dd70357...) | MIT (the repository's LICENSE at that commit) | rows with `source == custom` only (221 of 313; the others were collected from third-party sets), 8 per category = 48; fetched by `changegate fetch-data`, not shipped | `battery.safety-retention` | 2026-09-27 |
| `harmbench` | HarmBench behaviors, test split (Mazeika et al., arXiv:2402.04249) | centerforaisafety/HarmBench@8e1604d1 | github.com/centerforaisafety/HarmBench, `data/behavior_datasets/harmbench_behaviors_text_test.csv` (sha256 75d257b3...) | MIT (LICENSE at that commit) | `standard` behaviors only, 5 per semantic category = 30; fetched, not shipped | `battery.safety-retention` | 2026-09-27 |
| `xstest` | XSTest prompts (Röttger et al., arXiv:2308.01263, NAACL 2024) | paul-rottger/xstest@d7bb5bd7 | github.com/paul-rottger/xstest, `xstest_prompts.csv` (sha256 11783fb2...) | CC-BY-4.0; attribute Röttger et al. | safe prompts, 5 per type = 50; fetched, not shipped. The repository's model completions are under other terms and are not used | `battery.over-refusal` | 2026-09-27 |
| `emergent-misalignment` | EM evaluation questions (Betley et al., arXiv:2502.17424) | emergent-misalignment/emergent-misalignment@80c11967 | github.com/emergent-misalignment/emergent-misalignment, `evaluation/first_plot_questions.yaml` (sha256 215abde5...) | MIT (LICENSE at that commit) | the 8 plain questions or their `_json` variants; fetched, not shipped. The repository's training data files were not checked one by one, so the EM organism recipe stays blocked until they are | `battery.em` | 2026-09-27 |
| `synthetic` | Sycophancy factual pairs | this repository | `changegate.battery.itemsets` | Apache-2.0 | all: 20 pairs (neutral, and with a suggested wrong answer) | `battery.sycophancy` | 2026-09-27 |

## Attribution

- **XSTest** (CC-BY-4.0): Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio,
  Federico Bianchi and Dirk Hovy [R], "XSTest: A Test Suite for Identifying Exaggerated Safety
  Behaviours in Large Language Models", NAACL 2024, arXiv:2308.01263, from
  github.com/paul-rottger/xstest at commit d7bb5bd7, licensed under CC-BY-4.0
  (https://creativecommons.org/licenses/by/4.0/). changegate selects the safe prompts (5 per
  type) and changes nothing in their text. Any redistribution of the fetched items carries this
  line.

## Not recorded, so not loaded

| Dataset | Needed by | State |
|---|---|---|
| Emergent-misalignment insecure-code training set (Betley et al., arXiv:2502.17424) | `organisms.recipes.emergent` | License not read. The recipe downloads nothing; it reads a sealed local path and refuses unless `license_verified` is set. A row goes here before that flag is set for any build. |

## Checked and not used

| Dataset | Declared license | Checked | Reason |
|---|---|---|---|
| tatsu-lab/alpaca (Hugging Face dataset, revision dce01c9b) | cc-by-nc-4.0 | 2026-09-27, `scripts/fetch_prompts.py --check-license hf:tatsu-lab/alpaca` | Non-commercial, so not on the permissive allowlist. `drift.refusal` uses the synthetic harmless set instead, which departs from arXiv:2406.11717 (Alpaca harmless prompts). |
