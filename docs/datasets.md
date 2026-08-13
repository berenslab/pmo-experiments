# Datasets

Reference for the datasets registered in `pmo_experiments.datasets` (`DATASET_REGISTRY`). Every entry below is a registry key you can pass to `--datasets`/`--eval-datasets`/`DATASET.NAME`. Paths are the defaults for `DATASET_PATH` (raw data) and `CONVERTED_DATASET_PATH` (OpenCLIP-ready output); both are config overrides, so you don't have to use these exact locations.

## Training datasets overview

These are the sources compared in the paper (see `scripts/generate_configs.py:DEFAULT_DATASETS` for the exact sweep).

| Key | Source | Default raw path | Default converted path | Citation |
| --- | --- | --- | --- | --- |
| `deepeyenet` | DeepEyeNet retinal fundus images + medical reports | `datasets/deepeyenet` | `datasets/deepeyenet/converted` | Huang et al. 2021, WACV ([GitHub](https://github.com/Jhhuangkay/DeepOpht-Medical-Report-Generation-for-Retinal-Images-via-Deep-Models-and-Visual-Explanation)) |
| `kaggle_eyepacs` | Kaggle EyePACS diabetic retinopathy detection challenge | `datasets/kaggle-eyepacs/diabetic-retinopathy-detection.zip` | `datasets/kaggle-eyepacs/converted` | Dugas et al. 2015, [Kaggle](https://kaggle.com/competitions/diabetic-retinopathy-detection) |
| `flair` | FLAIR's assembled fundus datasets | `datasets/flair/raw` | `datasets/flair/converted` | Silva-Rodríguez et al. 2025, [Medical Image Analysis](https://doi.org/10.1016/j.media.2024.103357) |
| `pubmed_ophtha` | Full PubMed-Ophtha, panel-centric, train/val/test split | `datasets/pubmed_ophtha/pubmed_ophtha.parquet` | `datasets/pubmed_ophtha/converted` | this paper (see [Citation](../README.md#citation)) |
| `pubmed_ophtha_cfp` | PubMed-Ophtha restricted to color fundus photograph (CFP) panels | same parquet as `pubmed_ophtha` | `datasets/pubmed_ophtha_cfp/converted` | same as `pubmed_ophtha` |
| `pubmed_ophtha_cfp_matched` | `pubmed_ophtha_cfp` downsampled to DeepEyeNet's size (train 9428 / val 3142 / test 3140) | same parquet | `datasets/pubmed_ophtha_cfp_matched/converted` | same as `pubmed_ophtha` |
| `pubmed_ophtha_decon` | `pubmed_ophtha` with panels overlapping the evaluation datasets removed | same parquet | `datasets/pubmed_ophtha_decon/converted` | same as `pubmed_ophtha` |
| `pubmed_ophtha_in_text_mentions` | `pubmed_ophtha` with in-text mentions added as additional captions | same parquet | `datasets/pubmed_ophtha_in_text/converted` | same as `pubmed_ophtha` |
| `pubmed_ophtha_full` | `pubmed_ophtha` with a 5% val split and no test split, for maximum training data | same parquet | `datasets/pubmed_ophtha_full/converted` | same as `pubmed_ophtha` |

## Evaluation datasets overview

Used by `pmo_experiments eval run` for k-NN, zero-shot, and linear-probing evaluation. Not used for training.

| Key | Source | Default raw path | Default converted path | Citation |
| --- | --- | --- | --- | --- |
| `aptos2019` | APTOS 2019 Blindness Detection | `datasets/aptos2019/APTOS2019.zip` | `datasets/aptos2019/converted` | [Kaggle](https://kaggle.com/competitions/aptos2019-blindness-detection) |
| `brset` | BRSET (Brazilian ophthalmological dataset) | `datasets/brset` | `datasets/brset/converted` | Nakayama et al. 2024, [PLOS Digital Health](https://doi.org/10.1371/journal.pdig.0000454) |
| `eddfs` | EDDFS | `datasets/eddfs/OriginalImages.zip` | `datasets/eddfs/converted` | Xia et al. 2022, [MMSP](https://doi.org/10.1109/MMSP55362.2022.9949547) |
| `refuge2` | REFUGE2 (glaucoma detection) | `datasets/refuge2/REFUGE2.zip` | `datasets/refuge2/converted` | Fang et al. 2022, [arXiv](https://arxiv.org/abs/2202.08994) |
| `rfmid_1` | RFMiD v1 (retinal fundus multi-disease) | `datasets/rfmid_1` | `datasets/rfmid_1/converted` | Pachade et al. 2021, [Data](https://www.mdpi.com/2306-5729/6/2/14) |
| `rfmid_2` | RFMiD v2 (retinal fundus multi-disease) | `datasets/rfmid_2` | `datasets/rfmid_2/converted` | Panchal et al. 2023, [Data](https://www.mdpi.com/2306-5729/8/2/29) |
| `joint_rfmid` | RFMiD v1 + v2 combined, restricted to columns present in both (plus an "other" column for the rest) | via `RFMID_1_DATASET_PATH`/`RFMID_2_DATASET_PATH` prefixed args, forwarded to `rfmid_1`/`rfmid_2` | `datasets/joint_rfmid/converted` | see `rfmid_1`/`rfmid_2` |

Citations above are drawn from `util/dataset_decontamination.py:TEST_DATASET_CITATION_MAP`, the same metadata this repo uses internally to detect literature overlap with these datasets. See that module for full author lists and additional citations per dataset (several have more than one, e.g. BRSET and REFUGE/REFUGE2).

## Dataset details

### DeepEyeNet

Retinal fundus images paired with medical report text. No special setup beyond placing the raw data at `DATASET_PATH`.

### Kaggle EyePACS (`kaggle_eyepacs`)

`scripts/datasets/download_kaggle_eyepacs_dataset.sh` downloads the raw zip via the Kaggle API into `datasets/kaggle-eyepacs/diabetic-retinopathy-detection.zip` (the default `DATASET_PATH`). It requires a Kaggle API token, either as `~/.kaggle/kaggle.json` or the `KAGGLE_API_TOKEN` environment variable:

```bash
./src/pmo_experiments/scripts/datasets/download_kaggle_eyepacs_dataset.sh
```

### FLAIR (`flair`)

FLAIR itself supports ~30 public fundus datasets, but `FlairDataset` (`src/pmo_experiments/datasets/flair_dataset.py`) only uses this subset for training (see `selected_training_datasets`):

`01_EYEPACS`, `05_1000x39`, `07_LAG`, `08_ODIR-5K`, `09_PAPILA`, `10_PARAGUAY`, `11_STARE`, `12_ARIA`, `14_AGAR300`, `16_FUND-OCT`, `17_DiaRetDB1`, `18_DRIONS-DB`, `19_Drishti-GS1`, `20_E-ophta`, `21_G1020`, `23_HRF`, `24_ORIGA`, `26_ROC`, `28_OIA-DDR`, `29_AIROGS`, `30_SUSTech-SYSU`, `31_JICHI`, `32_CHAKSU`, `33_DR1-2`, `34_Cataract`.

You only need to download and prepare **this subset**, not all of FLAIR's datasets. Download links and preparation instructions for each one are in the [FLAIR README](https://github.com/jusiro/FLAIR), under "Foundation model pre-training". Follow its dataset table and the `flair_datasets.constants`/`flair_datasets.prepare_partitions` setup (installed automatically as part of the `flair[datasets]` dependency, see [README](../README.md#the-flair-dependency)) before pointing `DATASET.DATASET_PATH` at the prepared data.

`FlairDataset` does its own preprocessing step, so it uses `CONVERTED_FLAIR_DATASET_PATH` (default `datasets/flair/converted`) as a separate converted-path key, distinct from `CONVERTED_DATASET_PATH`.

### PubMed-Ophtha (`pubmed_ophtha`, `pubmed_ophtha_cfp`, `pubmed_ophtha_cfp_matched`, `pubmed_ophtha_decon`, `pubmed_ophtha_in_text_mentions`, `pubmed_ophtha_full`)

All `pubmed_ophtha*` variants read the same `pubmed_ophtha.parquet` file (see [Quickstart](../README.md#quickstart-training-on-pubmed-ophtha) for how it's obtained) and mainly differ in how `prepare_train_test_val_dfs` filters/splits it. The columns they filter on are `contains_cfp`, `contains_oct`, `contains_retinal`, `contains_other`, `contains_marked` (image-type flags used by the CFP/matched variants) and `article_id`/`panel_id` (used for the article-grouped stratified split and decontamination). `pubmed_ophtha_decon` additionally uses `util/dataset_decontamination.py` to remove panels overlapping the evaluation datasets. The full parquet schema is documented in the [`pubmed-ophtha`](https://github.com/berenslab/pubmed-ophtha) repo.

`to_open_clip_format()` downloads `pubmed_ophtha.parquet` from the [`pubmed-ophtha/PubMed-Ophtha`](https://huggingface.co/datasets/pubmed-ophtha/PubMed-Ophtha) Hugging Face dataset repo (via `huggingface_hub`'s `hf://` `fsspec` integration) the first time it's called, if the parquet isn't already present at `DATASET.DATASET_PATH` (default `datasets/pubmed_ophtha`) — so the first train/eval run needs internet access. To skip this, place your own copy of the parquet at that path beforehand.

`configs generate` leaves `DATASET.DATASET_PATH` as `null` in generated configs, which falls back to each dataset class's own default. That default is `datasets/pubmed_ophtha` for **every** `pubmed_ophtha_*` variant, since they all read the identical raw parquet from that one location by default. Only override `DATASET_PATH` (via the YAML or a CLI override, e.g. `DATASET.DATASET_PATH=/path/to/data`) if your parquet lives somewhere else. It's not required, but more efficient to use the **same** path for every variant in that case too, so you don't end up with redundant copies of the parquet file. `CONVERTED_DATASET_PATH` stays per-variant (its class default already differs per variant), since each variant produces different train/val/test CSVs; point `DATASET.IMAGES_FOLDER` at a shared location if you also want variants to reuse already-extracted panel images instead of re-extracting them from the parquet.

As described in the manuscript, for some figures the panel images and subcaptions are not available upon download due to license restrictions. These fields can be populated using the download scripts in the [`pubmed-ophtha`](https://github.com/berenslab/pubmed-ophtha) repo.

Panels with a missing `panel_image_bytes` (or an image that fails to decode) are silently dropped from the converted `train.csv`/`val.csv`/`test.csv` by `to_open_clip_format()`. They don't cause an error, they just don't end up in the training data. This happens after the train/val/test split is computed, so it doesn't affect split balance, but it does mean the resulting dataset is smaller than the raw panel count, with no summary of how many panels were dropped this way. To check how many panels in your local parquet are affected before training, run:

```python
import pandas as pd
df = pd.read_parquet("datasets/pubmed_ophtha/pubmed_ophtha.parquet", columns=["panel_image_bytes"])
print(df["panel_image_bytes"].isna().sum())
```

`pubmed_ophtha_decon` uses a bundled `contaminated_panels.json` by default, so no download is needed. Set `USE_CACHED_CONTAMINATION_RESULTS: "False"` to recompute it instead. This downloads full article text from PMC (`util/pmc_article_download.py`, throttled to 5 concurrent requests with a 2s delay each; slow for the full dataset). Bodies are cached under `ARTICLE_BODY_PATH` (default `datasets/pubmed_ophtha_decon/article_bodies`). Set `DELETE_ARTICLE_BODIES: "True"` to remove them afterward. The recomputed file is written to `CONVERTED_DATASET_PATH/contaminated_panels.json`.

### APTOS 2019 (`aptos2019`)

APTOS 2019 Blindness Detection. `DATASET_PATH` points at the downloaded `APTOS2019.zip`.

### BRSET (`brset`)

Brazilian ophthalmological dataset. `DATASET_PATH` points at the extracted dataset directory.

### EDDFS (`eddfs`)

`DATASET_PATH` points at the downloaded `OriginalImages.zip`.

### REFUGE2 (`refuge2`)

Glaucoma detection dataset. `DATASET_PATH` points at the downloaded `REFUGE2.zip`.

### RFMiD (`rfmid_1`, `rfmid_2`, `joint_rfmid`)

`rfmid_1`/`rfmid_2` expect `DATASET_PATH` to point at a directory with the dataset's original per-split folder layout, not a single file:

```
# rfmid_1 (DATASET_PATH, e.g. datasets/rfmid_1)
Training_Set/Training/<ID>.png            Training_Set/RFMiD_Training_Labels.csv
Evaluation_Set/Validation/<ID>.png        Evaluation_Set/RFMiD_Validation_Labels.csv
Test_Set/Test/<ID>.png                    Test_Set/RFMiD_Testing_Labels.csv

# rfmid_2 (DATASET_PATH, e.g. datasets/rfmid_2) - images sit next to the label CSV, no image subfolder
Train/<ID>.jpg             Train/RFMiD_2_Training_labels.csv
Validation/<ID>.jpg        Validation/RFMiD_2_Validation_labels.csv
Test/<ID>.jpg              Test/RFMiD_2_Testing_labels.csv
```

`joint_rfmid` combines both: pass `RFMID_1_DATASET_PATH`/`RFMID_2_DATASET_PATH` (and other `RFMID_1_*`/`RFMID_2_*`-prefixed args) instead of a single `DATASET_PATH`, and they're forwarded to the respective `rfmid_1`/`rfmid_2` instance internally.
