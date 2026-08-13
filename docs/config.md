# Configuration

Training configs are YAML files loaded with [OmegaConf](https://omegaconf.readthedocs.io/) (`pmo_experiments.configs.config.get_config`). `src/pmo_experiments/configs/model_configs/default_config.yaml` is the shipped default; `pmo_experiments configs generate` sweeps model x seed x dataset combinations into a tree of YAMLs (see [the pipeline overview](../README.md#pipeline)), generating them with the hyperparameters described in the paper's Methods section (architecture, epochs, batch size, LR schedule, seeds, etc.). See `scripts/generate_configs.py` if you need the exact values.

## Top-level sections

| Section | Purpose |
| --- | --- |
| `DATASET` | `NAME` (registry key, see [`docs/datasets.md`](datasets.md)), plus dataset-specific args like `DATASET_PATH`/`CONVERTED_DATASET_PATH` forwarded as constructor kwargs. Some datasets define extra keys of their own, e.g. `pubmed_ophtha_decon` and `flair`; see [`docs/datasets.md`](datasets.md) for the full list per dataset. |
| `MODEL` | `ARCHITECTURE` and `PRETRAINED_WEIGHTS` passed to OpenCLIP (see below) |
| `TRAINING` | Batch size, epochs, learning rate, precision, warmup, device count, etc. |
| `OUTPUT` | `SAVE_DIR`, `WANDB_PROJECT_BASE_NAME`, `RUN_NAME`, logging interval |
| `RESUME` / `GRADIENT_CHECKPOINTING` | Top-level training flags |
| `SEED` | Training seed |

Any value can be overridden from the CLI as a dotlist argument, e.g. `pmo_experiments train --config-path ... TRAINING.BATCH_SIZE_PER_GPU=512`.

## Composing configs

- `__BASE__: <path>` (or a list of paths) merges one or more base configs in before the current one, recursively; a config's `__BASE__` cannot point at itself or at the shipped default (use `MERGE_DEFAULT` for the latter).
- `MERGE_DEFAULT: true` additionally merges the config with `default_config.yaml` to fill in anything not set explicitly; values already present in your config take precedence.

A minimal hand-written config typically looks like:

```yaml
__BASE__: null
MERGE_DEFAULT: true
DATASET:
  NAME: "pubmed_ophtha"
```

## Model architecture and pretrained weights

`MODEL.ARCHITECTURE` selects the OpenCLIP model architecture (e.g. `ViT-B-16`), and `MODEL.PRETRAINED_WEIGHTS` selects which pretrained checkpoint to initialize it from before finetuning. Both are passed straight through to OpenCLIP's training entry point as `--pretrained` (`train_model.py` maps `MODEL.PRETRAINED_WEIGHTS` to it); if `PRETRAINED_WEIGHTS` is `null`, the model is trained from scratch (random init) instead of starting from a checkpoint.

Neither field is limited to the catalog shown by `pmo_experiments configs list-models`. Any OpenCLIP-compatible architecture/pretrained-tag pair works. See OpenCLIP's [pretrained model list](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md) for the built-in `ARCHITECTURE`/`PRETRAINED_WEIGHTS` combinations (e.g. `ARCHITECTURE: ViT-B-16`, `PRETRAINED_WEIGHTS: openai`), and its [Hugging Face Hub loading docs](https://github.com/mlfoundations/open_clip#loading-models) for the `hf-hub:` syntax used to load arbitrary Hub backbones such as BiomedCLIP directly as the architecture, with `PRETRAINED_WEIGHTS: null` since the Hub checkpoint already provides the weights:

```yaml
MODEL:
  ARCHITECTURE: hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
  PRETRAINED_WEIGHTS: null
```

The model catalog (`scripts/generate_configs.py:DEFAULT_MODELS`/`MODEL_TO_GPU_COUNT_MAP`) just supplies convenient defaults (e.g. GPU count) for `configs generate`'s sweep. It isn't enforced at train time.
