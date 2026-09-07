# pmo-experiments

[![Pre-commit](https://github.com/berenslab/pmo-experiments/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/berenslab/pmo-experiments/actions/workflows/pre-commit.yml)
[![Linting](https://github.com/berenslab/pmo-experiments/actions/workflows/linting.yml/badge.svg)](https://github.com/berenslab/pmo-experiments/actions/workflows/linting.yml)
[![Spell Checking](https://github.com/berenslab/pmo-experiments/actions/workflows/spell_checking.yml/badge.svg)](https://github.com/berenslab/pmo-experiments/actions/workflows/spell_checking.yml)
[![Static Type Checking](https://github.com/berenslab/pmo-experiments/actions/workflows/static_type_checking.yml/badge.svg)](https://github.com/berenslab/pmo-experiments/actions/workflows/static_type_checking.yml)

CLIP finetuning, evaluation, and plotting pipeline for **"Scientific Domain Knowledge Improves Vision-Language Fundus Models"**. See [Resources](#resources) for the other repositories and released artifacts behind the paper.

## Resources

|                       |                                                                                                             |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| Preprint              | [arXiv:2605.02720](https://arxiv.org/abs/2605.02720)                                                        |
| Dataset               | [pubmed-ophtha/PubMed-Ophtha](https://huggingface.co/datasets/pubmed-ophtha/PubMed-Ophtha)                  |
| Dataset pipeline      | [berenslab/pubmed-ophtha](https://github.com/berenslab/pubmed-ophtha)                                       |
| PDF parser            | [berenslab/pmo-parser](https://github.com/berenslab/pmo-parser)                                             |
| CLIP experiments      | *This repository*                                                                                           |
| Figure-parsing models | [pubmed-ophtha/detection-models](https://huggingface.co/pubmed-ophtha/detection-models)                     |
| PubMed-Ophtha CLIP    | [PubMed-Ophtha CLIP Models](https://huggingface.co/collections/pubmed-ophtha/pubmed-ophtha-clip-models)     |
| Paper checkpoints     | [pubmed-ophtha/experiment-checkpoints](https://huggingface.co/pubmed-ophtha/experiment-checkpoints)         |

This README focuses on this repo's own pipeline; see the linked repos for how the dataset itself is built.

> **Getting started:** [Installation](#installation) → [Quickstart: training on PubMed-Ophtha](#quickstart-training-on-pubmed-ophtha). The dataset downloads itself on first use, so those two sections are all you need to train a model. See [Pipeline](#pipeline) for the end-to-end command sequence.

## Contents

- [Resources](#resources)
- [Results: linear probing](#results-linear-probing)
- [Installation](#installation)
- [Quickstart: training on PubMed-Ophtha](#quickstart-training-on-pubmed-ophtha)
- [Pipeline](#pipeline)
- [Configuration](#configuration)
- [Evaluation & plotting](#evaluation--plotting)
- [Model weights](#model-weights)
- [Development](#development)
- [The FLAIR dependency](#the-flair-dependency)
- [Citation](#citation)
- [License](#license)

## Results: linear probing

Mean AUROC (%, higher is better) across all tasks in each task group; **bold** = best, *italic* = second-best. The average column covers all 110 evaluation tasks.

| Dataset/Model | Disease risk | AMD | Glaucoma | Myopia | HR | AH | MS | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kaggle EyePACS | 92.47 | 82.51 | 77.09 | 92.98 | 77.48 | 91.04 | 77.57 | 80.00 |
| FLAIR | *93.33* | 89.16‡ | **87.38**‡ | 94.93‡ | 78.74‡ | 92.02‡ | 82.05‡ | 83.58 |
| DeepEyeNet | 93.32 | **94.69** | 84.86 | *98.66* | *78.92* | *97.64* | *87.90* | *85.68* |
| PubMed-Ophtha | **93.83** | *93.81* | *86.35* | **98.76** | **82.10** | **98.16** | **88.90** | **88.63** |
| BiomedCLIP (baseline) | 90.80 | 83.96 | 78.73 | 97.85 | 71.88 | 92.91 | 82.44 | 83.32 |

AMD: age-related macular degeneration, HR: hypertensive retinopathy, AH: asteroid hyalosis, MS: macular scars.

‡ FLAIR's training data directly overlaps the classification labels for AMD, Glaucoma, Myopia, HR, AH, and MS, so those scores reflect in-domain classification rather than generalization to unseen labels.

These are the paper's numbers, averaged over three seeds and trained on the standard
PubMed-Ophtha split. The single checkpoints released in
[PubMed-Ophtha CLIP Models](https://huggingface.co/collections/pubmed-ophtha/pubmed-ophtha-clip-models)
were trained on the larger `pubmed_ophtha_full` split and score slightly higher; the checkpoints
behind the table below are in
[pubmed-ophtha/experiment-checkpoints](https://huggingface.co/pubmed-ophtha/experiment-checkpoints).

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (**recommended**):

```bash
uv sync                      # base dependencies (requires Python >= 3.12)
uv sync --extra plots        # + matplotlib/seaborn/opentsne, needed for `pmo_experiments plots`
uv sync --group dev          # + linting/type-checking tools, for development
```

The training framework is a fork of OpenCLIP (`verena-hallitschke/open_clip@feature/add-text-augmentations`, adds text-augmentation support), pulled automatically via `[tool.uv.sources]` in `pyproject.toml`. No separate install step is needed.

Alternatively, install with `pip` into a Python >= 3.12 environment:

```bash
pip install .                                       # base dependencies
pip install ".[plots]"                              # + matplotlib/seaborn/opentsne
```

Training always reports metrics to [Weights & Biases](https://wandb.ai/) (`--report-to wandb` is hardcoded, not config-controlled), so `pmo_experiments train` needs a W&B login before it will run: `wandb login`, or set `WANDB_API_KEY`. To run without a W&B account, set `WANDB_MODE=offline` (or `disabled`) instead.

## Quickstart: training on PubMed-Ophtha

1. **Get the data.** `pubmed_ophtha.parquet` is downloaded automatically from [Hugging Face](https://huggingface.co/datasets/pubmed-ophtha/PubMed-Ophtha) the first time it's needed (i.e. on your first `train`/`eval run`), if it isn't already present. It's saved to `pubmed_ophtha.parquet` under `DATASET.DATASET_PATH`, which defaults to `datasets/pubmed_ophtha`. This requires internet access on that first run. Use this same directory for every `pubmed_ophtha_*` variant you train on (see [`docs/datasets.md`](docs/datasets.md)); they all read the identical parquet file. If you'd rather stage the file yourself (e.g. for offline use), download it manually and place it at that same path to skip the automatic download.

   As described in the paper, for some figures the panel images and subcaptions are not available upon download due to license restrictions. These fields can be populated using the download scripts in the [`pubmed-ophtha`](https://github.com/berenslab/pubmed-ophtha) repo.

2. **Generate a config:**
   ```bash
   pmo_experiments configs generate --datasets pubmed_ophtha --models ViT-B-16 --seeds 106080954
   ```
   This writes `generated_configs/106080954/pubmed_ophtha/config_ViT-B-16_bs2048.yaml` with `DATASET.DATASET_PATH: null`, which falls back to the dataset class's own default path (`datasets/pubmed_ophtha`) at train time. If your data lives elsewhere, override it (edit the YAML, or pass it as a CLI override, as in step 3 below). Use `pmo_experiments configs list-datasets` to see all registered dataset keys. You can also hand-write a minimal config instead, using the shipped default as a base:
   ```yaml
   __BASE__: null
   MERGE_DEFAULT: true
   DATASET:
     NAME: "pubmed_ophtha"
   ```

3. **Train:**
   ```bash
   pmo_experiments train --config-path generated_configs/106080954/pubmed_ophtha/config_ViT-B-16_bs2048.yaml
   ```
   Resume from a checkpoint with `--resume-from <path>.pt`, or override any config value inline, e.g. `TRAINING.BATCH_SIZE_PER_GPU=512` or `DATASET.DATASET_PATH=/path/to/data`.

Under the hood, the dataset's `to_open_clip_format()` first materializes an OpenCLIP-ready CSV/image (or WebDataset) layout from the raw parquet. Training then launches OpenCLIP (`open_clip_train.main`) as a subprocess, single-GPU directly or via `torchrun` for multi-GPU architectures.

## Pipeline

```
1. configs generate   → sweep model x seed x dataset configs
2. train <config>     → finetune a CLIP model
3. eval run           → k-NN / zero-shot / linear-probing eval; run once per split (see below)
4. eval aggregate     → collapse raw eval outputs into per-target CSVs
5. plots auroc/winrate/tables → render the paper's figures/tables from the aggregated CSVs
```

Run any subcommand with `--help` for the full flag list, e.g. `pmo_experiments eval run --help`.

## Configuration

Training configs are YAML files with sections for `DATASET`, `MODEL`, `TRAINING`, and `OUTPUT`, composable via `__BASE__`/`MERGE_DEFAULT`, and overridable from the CLI as dotlist args. This also covers how to point `MODEL.ARCHITECTURE` at an arbitrary OpenCLIP-compatible or Hugging Face Hub backbone (e.g. BiomedCLIP) instead of the built-in model catalog. See [`docs/config.md`](docs/config.md) for the full reference.

## Evaluation & plotting

`eval run` must be run once per split. Both are required, not just `test`:

```bash
pmo_experiments eval run --eval-split val     # used to select linear-probing hyperparameters
pmo_experiments eval run --eval-split test    # final reported scores
pmo_experiments eval aggregate
pmo_experiments plots auroc
pmo_experiments plots winrate
pmo_experiments plots tables
```

`eval run` discovers trained model folders (`clip_models_*`) and evaluates each against the requested datasets. `eval aggregate` writes aggregated per-target CSVs (e.g. `evaluation_results/aggregated_scores/final_test_scores_auroc.csv`). `plots auroc`/`plots winrate` build the paper's figures from those CSVs. `plots tables` builds the paper's task group and PubMed-Ophtha-variant CSV tables (one per eval setting) from the same CSV, written to `eval_tables/` by default.

## Model weights

All checkpoints behind the paper's figures and tables — five vision encoder architectures ×
five training data sources × three seeds, plus the PubMed-Ophtha variants — are released at
[pubmed-ophtha/experiment-checkpoints](https://huggingface.co/pubmed-ophtha/experiment-checkpoints),
with a `MANIFEST.csv` mapping each path to its architecture, training data source, seed, and mean
AUROC per evaluation setting.

For the three curated single models intended for downstream use, see
[PubMed-Ophtha CLIP Models](https://huggingface.co/collections/pubmed-ophtha/pubmed-ophtha-clip-models).
These were trained on the larger `pubmed_ophtha_full` split rather than the paper's split — use the
experiment checkpoints to reproduce the paper.

## Development

```bash
uv sync --group dev
pre-commit install
```

Pre-commit runs ruff (lint + format), pyright, detect-secrets, and `codespell`; a separate CI workflow additionally runs [`typos`](https://github.com/crate-ci/typos) against `_typos.toml`. See `.pre-commit-config.yaml` and `ruff.toml` for the exact configuration.

To silence a false-positive `typos` finding inline, use one of the directives configured in `_typos.toml`:
- `# spellchecker:disable-line` at the end of a line: ignores that line.
- `# spellchecker:ignore-next-line`: ignores the line that follows it.
- `# spellchecker:off` / `# spellchecker:on`: ignores everything in between.

## The FLAIR dependency

The `flair` package (installed automatically) is a fork of [jusiro/FLAIR](https://github.com/jusiro/FLAIR) (Silva-Rodríguez et al., *Medical Image Analysis*, 2024), Apache-2.0 licensed. All credit for FLAIR itself goes to the original authors. The fork packages FLAIR's local-data preparation and default-weight loading as an installable library (with the `local_data` code split out into the `flair_datasets` extra) so they can be called programmatically from this repo's dataset classes, instead of run as FLAIR's own standalone scripts. The modeling code is unchanged.

Using the `flair` dataset still requires downloading and preparing FLAIR's sub-datasets yourself, following the instructions and dataset table in the [FLAIR README](https://github.com/jusiro/FLAIR). See [`docs/datasets.md`](docs/datasets.md) for exactly which subset of FLAIR's datasets this repo trains on.

## Citation

```bibtex
@misc{hallitschke2026scientific,
      title={Scientific Domain Knowledge Improves Vision-Language Fundus Models},
      author={Verena Jasmin Hallitschke and Carsten Eickhoff and Philipp Berens},
      year={2026},
      eprint={2605.02720},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.02720}, 
}
```

## License

This repository's own code is [MIT licensed](LICENSE). The `flair` dependency (installed from [verena-hallitschke/FLAIR](https://github.com/verena-hallitschke/FLAIR)) is licensed separately under Apache-2.0 by its original authors; its `LICENSE`/`NOTICE` ship in the installed package's dist-info.
