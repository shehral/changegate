# Credits

changegate reimplements published techniques. Each row names the source, what changegate
uses from it, and where. Add a row in the same change that adds the technique. Titles and
author lists should be copied from the arXiv abstract page when a row is first used in code.
Rows marked [R] give authors or venue from memory; check them at the source before any
release.

| Source | Used for | Where |
|---|---|---|
| Arditi et al., [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) | Refusal is mediated by a single residual-stream direction; changegate estimates the base's refusal direction from contrast prompts and measures projections and cosines against it | `lineage`, `adapter-diff`, `drift` |
| Hurtado, [arXiv:2607.01854](https://arxiv.org/abs/2607.01854) | Auditing models for abliteration (refusal-direction removal); informs the abliteration ratio and its known failure against spoofed references and white-box adversaries | `lineage` |
| Chen et al., [arXiv:2507.21509](https://arxiv.org/abs/2507.21509) | Persona vectors: trait directions whose shift after fine-tuning tracks trait change; changegate builds a prompt-only variant from contrastive system prompts (no generation, no judge) and reports the fine-tuning shift along the base's trait direction | `drift.persona`; `adapter-diff` (persona directions projected, reported, not scored) |
| [arXiv:2606.15980](https://arxiv.org/abs/2606.15980) | Probe and monitor staleness after weight changes, staleness prediction and label-free realignment | monitor revalidation (later) |
| Bhardwaj et al., [arXiv:2402.11746](https://arxiv.org/abs/2402.11746) | RESTA: re-adding a safety vector to a fine-tuned model; a repair recipe whose target metric the record keeps out of its re-verification endpoints | record (repair provenance) |
| Hsu et al., [arXiv:2405.16833](https://arxiv.org/abs/2405.16833) | Safe LoRA: projecting LoRA updates onto an aligned-minus-base subspace | not implemented; cited as related work (a possible repair recipe, later). `adapter-diff` does not use it |
| Betley et al., [arXiv:2502.17424](https://arxiv.org/abs/2502.17424) | Emergent misalignment from narrow fine-tuning; the evaluation questions behind the EM battery; the insecure-code organism | `battery.em`, `organisms.recipes.emergent` |
| Qi et al., [arXiv:2310.03693](https://arxiv.org/abs/2310.03693) | Fine-tuning, including on benign data, erodes safety behavior; the harmful-example positive control | `battery.safety-retention`, `organisms.recipes.qi_harmful` |
| He et al., [arXiv:2404.01099](https://arxiv.org/abs/2404.01099) | Benign-looking fine-tuning data that erodes safety; the benign-data positive control | `battery.safety-retention`, `organisms.recipes.benign_erosion` |
| Yan et al., [arXiv:2011.09161](https://arxiv.org/abs/2011.09161) | Negative flips: items a model update gets wrong that the old model got right, even when aggregate accuracy holds | `battery.report`, `gate` (flips beside the base-vs-base noise floor), record |
| Hong et al. [R], [arXiv:2403.15447](https://arxiv.org/abs/2403.15447) | Trustworthiness of compressed models: 4-bit holds up, 3-bit degrades; the quantized-variant control | `organisms.recipes.quantize` |
| Egashira et al. [R], [arXiv:2405.18137](https://arxiv.org/abs/2405.18137) | Quantization can be exploited to change a model's behavior; evidence for the quantized-variant control | `organisms.recipes.quantize` (motivation only) |
| Wang et al. [R], [arXiv:2506.19823](https://arxiv.org/abs/2506.19823) | Persona features mediate emergent misalignment; cited as a stated limit of the refusal-direction projection | `adapter-diff` (docstring limit) |
| Hu et al. [R], [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | LoRA: the low-rank update B·A and its alpha/r scaling | `adapter-diff`, `lineage` (adapter path), `organisms.recipes.lora_finetune` |
| Kalajdzievski [R], [arXiv:2312.03732](https://arxiv.org/abs/2312.03732) | rsLoRA: the alpha/sqrt(r) scaling | `adapter-diff`, `lineage` (adapter path) |
| Frantar et al. [R], [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) | GPTQ: the 3-bit quantized variant | `organisms.recipes.quantize` |
| Dettmers et al. [R], [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) | NF4 4-bit quantization (via bitsandbytes) | `organisms.recipes.quantize` |
| Souly et al., [arXiv:2402.10260](https://arxiv.org/abs/2402.10260) | StrongREJECT: the harmful-prompt items (custom-source rows) | `battery.safety-retention` |
| Mazeika et al., [arXiv:2402.04249](https://arxiv.org/abs/2402.04249) | HarmBench: the standard behaviors used as harmful-prompt items | `battery.safety-retention` |
| Röttger et al., [arXiv:2308.01263](https://arxiv.org/abs/2308.01263) | XSTest: the safe prompts for over-refusal, and the string-match approach to refusal scoring | `battery.over-refusal`, the refusal rubric |
| Zou et al., [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) | AdvBench harmful behaviors (the harmful contrast set) and the refusal prefixes | `directions` (harmful set), the refusal rubric |
| Sharma et al., [arXiv:2310.13548](https://arxiv.org/abs/2310.13548) | The "I think the answer is X, but I'm not sure" sycophancy format | `battery.sycophancy` |
| Llama Guard 3 and Qwen3Guard-Gen model cards | The output formats the local guard judge parses | `battery.judges` |
| The Hugging Face Hub model tree (`base_model` card metadata) | Discovery of a base's public derivatives | `census.hub` |

## Specifications

| Specification | Used for |
|---|---|
| [in-toto Attestation Framework](https://github.com/in-toto/attestation), Statement v1 | the record's outer structure (`_type`, `subject`, `predicateType`, `predicate`) |
| [DSSE](https://github.com/secure-systems-lab/dsse) (Dead Simple Signing Envelope) | the signed envelope around a record, and the offline verifier |
| [CycloneDX](https://cyclonedx.org/) ML-BOM, 1.7 | the model-identity BOM exported from a verified record (`changegate record bom`) |

## Statistics

| Source | Used for | Where |
|---|---|---|
| Wilson (1927) [R] | score interval for a proportion | `battery.stats` |
| Newcombe (1998) [R] | interval for a difference of independent proportions | `battery.stats` |
| Bonett and Price (2012), J. Educ. Behav. Stat. 37:479 [R] | interval for a difference of paired proportions | `battery.stats` |
| Cohen (1960) [R] | kappa for judge-human agreement | `battery.agreement` |
| Clopper and Pearson (1934), Biometrika 26(4):404-413 [R] | exact binomial interval | `census.stats` |
| Hanley and McNeil (1982), Radiology 143(1):29-36 [R] | AUROC as the Mann-Whitney U statistic | `scripts/eval_lineage.py` |
| Hyndman and Fan (1996), The American Statistician 50(4):361-365 [R] | quantile type 7 | `census.stats`, `envelope.calibrate` |
