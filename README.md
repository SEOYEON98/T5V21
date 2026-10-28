# T5V21: LSTM-based PM2.5 forecasting model

Python source code and technical documentation supporting a study submitted to *Environmental Modelling & Software*.

## Overview

T5V21 is the identifier for a long short-term memory (LSTM) experiment developed at the National Institute of Environmental Research (NIER), Republic of Korea, for forecasting fine particulate matter (PM2.5). `T5` identifies the five-year training period (2016–2020), and `V21` identifies the 2021 validation period. **T5V21 is an experiment identifier, not a semantic software version number.** The historical label `FASAP` appears in legacy files but is not used here as the formal release name.

The current code package focuses on the `SU` region, the `09h` forecast configuration, and PM2.5 prediction. It provides input preparation, model training, saved-model inference and evaluation scripts. The researchers report that the original implementation has been run successfully to produce predictions and evaluation results.

## Software information

| Field | Information |
| --- | --- |
| Name | T5V21 |
| Developer | SEOYEON KIM (KIM SEOYOEN), first author |
| Affiliation | National Institute of Environmental Research (NIER), Republic of Korea |
| Contact | ksy0720@korea.kr |
| Source code | https://github.com/SEOYEON98/T5V21 |
| Date first publicly available | 27 October 2026 (as reported by the repository owner) |
| Programming language | Python |
| Original Python environment | Python 3.5.2 |
| Deep-learning framework | TensorFlow 1.8.0 |
| CUDA toolkit reported for development | CUDA 9.0.176 |
| Other Python dependencies | NumPy, pandas, SciPy, scikit-learn; additional imports are defined in the source scripts |
| Software documentation | This README and the included source-code comments |
| Code access cost | The public GitHub repository can be browsed and downloaded without charge |
| Software licence | MIT License (see [`LICENSE`](LICENSE)) |

The original model was developed in a Linux-based computing environment. The operating system and GPU model used for the original training have not been documented. The configuration above is the **researcher-reported development environment**, not a claim that the source has been retested on the current server or on every platform.

## Repository structure

```text
T5V21/
├── Code/
│   ├── 01_DataBaseExtract.py       # Source-data extraction and preprocessing
│   ├── 02_TrainExecutive.py        # Training/data-preparation controller
│   ├── 03_TrainCode.py             # Model training and checkpoint generation
│   ├── 04_Evaluation.py            # Prediction evaluation
│   ├── 05_Performance.py           # Prediction using stored model weights
│   └── Module/
│       ├── Config.py               # Paths, regions and model configuration
│       ├── Model.py                # Encoder–decoder LSTM model
│       └── Utils.py                # Data preparation and evaluation utilities
├── README.md
├── LICENSE                          # MIT License for source code
├── DATA_LICENSE.md                 # CC BY 4.0 terms for release data/results
├── Read.txt                        # Legacy project description
└── CHANGELOG_EMS.md                # Explanation of year-setting changes
```

This repository provides the **source-code package**. The required input data, pretrained checkpoints and evaluation outputs are distributed separately as GitHub Release assets (`T5V21_Data.zip` and `T5V21_Result.zip`). The research team has supplied the GitHub Release link for these archives: [`data-v1`](https://github.com/SEOYEON98/T5V21/releases/tag/data-v1).

## Model and experiment configuration

- **Method:** Encoder–decoder LSTM with two LSTM layers and a fully connected prediction layer.
- **Hidden units:** 96 in each layer (as specified in the supplied model code).
- **Objective and optimizer:** RMSE-based training objective with Adam optimization.
- **Training period:** 1 January 2016–31 December 2020.
- **Validation period:** 1 January 2021–31 December 2021.
- **Forecast setup:** 09h configuration, with day groups D0, D1 and D2.
- **Inputs:** Observed pollutant and meteorological data, numerical forecast variables and cluster-associated environmental predictors.
- **Target:** PM2.5 concentration.

In the supplied 09h time-profile definition, the sequence contains 15 six-hour segments (`T01`–`T15`): `T01`–`T03` correspond to the preceding day, `T04`–`T07` to the forecast day, `T08`–`T11` to the following day, and `T12`–`T15` to two days later. The scripts use `T01`–`T04` as the observation-designated segments, and the forecast targets are grouped as D0 (`T05`–`T07`), D1 (`T08`–`T11`), and D2 (`T12`–`T15`). Detailed temporal-index handling is defined in `Module/Config.py` and `Module/Utils.py`.

### Notes on the training-date correction

The release scripts have been adjusted to match the experiment description supplied by the authors:

- `Code/02_TrainExecutive.py`: training-period feature screening uses `20160101`–`20201231`.
- `Code/03_TrainCode.py`: training uses `20160101`–`20201231`; the held-out split uses `20210101`–`20211231`.

The held-out split is still called `test_date` in the legacy script although it corresponds to **validation in 2021**. The original archive contained later date settings; see `CHANGELOG_EMS.md` for the changes. **Editing date settings does not retroactively update prepared data, scaling factors, stored model weights or previously generated results.** The provenance of any separately distributed pretrained artifacts should be checked before claiming exact numerical reproducibility of the manuscript results.

The independent prediction/evaluation examples currently present in the scripts may use later years (including 2024); these periods are separate from the model's training/validation split.

## Data description

The data provider is **NIER**. The model uses air-quality and meteorological observations, forecast variables and derived cluster-related inputs. The original working archive contained records in text/tabular form, a prepared Python pickle file, season/day-specific normalization information, and model outputs.

| Group | Example fields | Meaning and documented units |
| --- | --- | --- |
| Observed air quality | `O_PM2_5`, `O_PM10`, `O_NO2`, `O_O3`, `O_SO2`, `O_CO` | Observed pollutant concentrations; PM2.5 is expressed in µg/m³ and NO₂ in ppb |
| Observed meteorology | `O_ta`, `O_td`, `O_U`, `O_V`, `O_RH`, `O_RN_ACC` | Temperature, dew point, wind components, humidity and accumulated precipitation; temperature is reported in K |
| Forecast air quality/meteorology | `f_PM2_5`, `f_NO2`, `f_Ta`, and other `f_` fields | Forecast model predictors corresponding to air-quality and meteorological variables |
| Cluster-associated fields | `cs_850hpa_Ta`, `cs_850hpa_U`, `cs_850hpa_V`, `cs_850hpa_gpm`, `cs_700hpa_V` | `cs_` denotes **cluster**; associated upper-air meteorological predictors |
| Cluster-pattern fields | `cluster_pattern_P01`–`cluster_pattern_P05` | Five atmospheric transport pattern predictors described below |
| Target | `PM2_5` | Target PM2.5 concentration (µg/m³) |

The following five pattern interpretations were supplied by the research team:

| Pattern | Interpretation |
| --- | --- |
| P01 | Stagnant/coastal airflow associated with inflow from the Yellow Sea coast |
| P02 | Short-range direct transport over the Yellow Sea |
| P03 | Transport originating from Asian dust source regions |
| P04 | Transport from eastern China, between the Shandong and Shanghai areas |
| P05 | Northerly transport from the Manchurian region |

Unit examples above were provided by the researchers; the precise units for other fields and the calculation/encoding of derived cluster variables are not specified by this release. Preprocessing and handling of flagged values such as `-999` and `-9999` are implemented in the source. Input selection and Min–Max normalization procedures are implemented in `Module/Utils.py`; per-model feature and scaling metadata were included in the original working archive.

## Obtaining the data and running the code

**Data availability:** The T5V21 input data and associated pretrained-model and result archives are distributed separately through the GitHub Release `data-v1`.

- **Source code:** https://github.com/SEOYEON98/T5V21
- **Data and model/result release:** https://github.com/SEOYEON98/T5V21/releases/tag/data-v1
- **Research dataset:** `T5V21_Data.zip`
- **Pretrained-model and evaluation results:** `T5V21_Result.zip`
- **Software license:** MIT License ([`LICENSE`](LICENSE))
- **Data and associated model/result archive license:** Creative Commons Attribution 4.0 International ([`DATA_LICENSE.md`](DATA_LICENSE.md))

Users can obtain the code from the main repository and access the data and result archives from the linked release. Reuse is permitted without individual permission, subject to the applicable licence's attribution and notice requirements. The links and files should be checked for public download access after publishing the release.

A prepared data file in the original working archive,
`Data/ConcatData/data_09h.pickle`, is approximately 518 MiB in
uncompressed form and exceeds standard Git file size limits; the release
archive allows data to be distributed separately from source code.
Do not load pickle files from untrusted sources.

To run the code after obtaining the required data and checkpoints:

1. Set working-directory paths and required input/output directories in `Code/Module/Config.py` and correct any legacy absolute imports in `Code/Module/Utils.py`.
2. Use a compatible Python 3.5.2 / TensorFlow 1.8.0 environment; configure CUDA appropriately if running with a GPU. Other dependency versions must be selected for compatibility with this historical environment.
3. Ensure that the prepared data, input-scaling metadata and trained model checkpoints exist under the paths expected by the scripts.
4. From the `Code` directory, review the output paths and selected evaluation year, then run the appropriate entry point:

```bash
# Prediction with pretrained model weights (requires external data/checkpoints)
python 05_Performance.py

# Evaluation of prediction tables (requires generated prediction tables)
python 04_Evaluation.py

# Training controller (requires appropriate historical data and configuration)
python 02_TrainExecutive.py
```

The prediction script may overwrite an existing output file. Run it using a working copy. The legacy configuration contains institution-specific absolute file paths and device selection; these must be adapted on other systems. The authors report successful prediction and evaluation in their computational environment; **this newly dated code-only distribution has not been independently executed in a clean environment**.

## Evaluation outputs

The existing result-table format uses `Date`, `Time`, `Target`, `AI`, and `CMAQ` fields. The evaluation script computes numerical performance measures such as RMSE, correlation (`R`), index of agreement (`IOA`) and normalized mean bias (`NMB`), together with category-based scores. Numerical matching to the manuscript depends on using the same data, model checkpoints, forecast period and preprocessing.

## Licence and reuse

The **Python source code and associated software documentation** are provided under the [MIT License](LICENSE). The separately distributed **research data and associated model/result artifacts** are provided under [CC BY 4.0](DATA_LICENSE.md).

Users do not need to request individual permission to use, modify or
redistribute material covered by these licences, provided that they
observe the respective licence conditions. Users of the data must
attribute NIER and indicate changes where applicable. Neither licence
constitutes an endorsement by NIER.

The copyright-holder designation in the MIT `LICENSE` file is **National Institute of Environmental Research (NIER)**. Reuse conditions for the dataset and model/result archives are specified in `DATA_LICENSE.md`.

## Citation and manuscript availability statement

Software: **T5V21**, developed by SEOYEON KIM, National Institute of Environmental Research (NIER), Republic of Korea. Source code: https://github.com/SEOYEON98/T5V21 (public repository, 27 October 2026). Contact: ksy0720@korea.kr.

Data and model/result release: https://github.com/SEOYEON98/T5V21/releases/tag/data-v1. For the *Environmental Modelling & Software* manuscript, include a separate bulleted **Software and Data Availability** subsection after the Highlights. This README does not substitute for the journal manuscript section.
