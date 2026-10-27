# LSTM-based PM2.5 forecasting: T5V21 experiment

**Software and data documentation for a manuscript submitted to *Environmental Modelling & Software* (EMS).**

> **Status (27 October 2026):** This README documents the `T5V21` distribution with the corrected date settings in `02_TrainExecutive.py` and `03_TrainCode.py`, and incorporates information supplied by the NIER researchers. The researchers report successful code execution, PM2.5 prediction and subsequent evaluation; no independent fresh-environment test was performed for this revised package. Public repository URLs, licence, full dependency pins and some variable metadata remain to be completed. **Changing script dates does not change previously saved checkpoints or prepared data.**

## 1. Overview

The archive contains Python code for an encoder–decoder long short-term memory (LSTM) model used to predict particulate matter with an aerodynamic diameter of 2.5 µm or less (PM2.5). The supplied experiment focuses on the region identified in the source code as `SU` and the `09h` forecast configuration. It includes source scripts, original and processed data, scaling metadata, pretrained TensorFlow checkpoints, and prediction/evaluation tables.

The research team identifies this software/experiment as **T5V21**. The original `Read.txt` also contains **FASAP** (*Forecasting AI System of Air-quality Pollutant*), which is preserved as a historical code label rather than presented here as the official released name.

**`T5V21` denotes the experiment configuration, not a numerical release version.** `T5` denotes training on 2016–2020 (five years) and `V21` denotes validation on 2021. The training scripts in this revised package have been aligned to those years; see Section 5 for the original discrepancy and retained artifacts.

### Software and release information

| Item | Information |
| --- | --- |
| Software / experiment name | **T5V21** – LSTM-based PM2.5 forecasting code |
| Experiment definition | `T5` = training in 2016–2020; `V21` = validation in 2021 |
| Historical source label | `FASAP` in `Read.txt` and legacy code comments; not the declared release name |
| Software release version | **[TO ADD actual repository release/tag; T5V21 alone is not a version]** |
| Developer | **KIM SEOYOEN** (first author and developer; English spelling supplied by the research team) |
| Affiliation | National Institute of Environmental Research (NIER), Republic of Korea |
| Contact | **KIM SEOYOEN**, `ksy0720@korea.kr` |
| Date first available | **[TO ADD actual date the repository is made public]** |
| Programming language | **Python 3.5.2** (original reported development environment) |
| Deep-learning framework | **TensorFlow 1.8.0** (original reported development environment) |
| CUDA | **9.0.176** (original reported development environment) |
| Operating system and GPU used for original training | **[TO CONFIRM]** |
| Other Python dependencies and versions | NumPy, pandas, SciPy, scikit-learn, etc.; **[TO ADD actual versions]** |
| Source-code repository | **[TO ADD public GitHub or equivalent URL]** |
| Archived version / DOI | **[TO ADD if issued]** |
| Software licence | **[TO ADD NIER-approved licence and `LICENSE` file]** |
| Source-code use cost | **[TO CONFIRM actual conditions at release]** |
| Dataset repository | **[TO ADD public URL/DOI, version, licence and access conditions]** |
| Documentation | This `README.md`; a data dictionary is recommended for detailed variable metadata |
| Source code size | About **69.3 KiB** of `.py` files in the originally supplied archive; revised size may differ slightly |
| Original archive size | About **319 MiB** compressed, **703 MiB** uncompressed including data and model results |

The source code is reported to have been developed by an NIER in-house researcher. The research team indicated code publication is permitted and training/validation datasets are shareable; the precise conditions governing raw data, pretrained weights, derived products and third-party reuse must still be documented.

## 2. Repository contents

The structure below represents the supplied archive; the revised package also contains this `README.md` and a change note. It has not yet been deposited in a verified public repository.

```text
T5V21/
├── Code/
│   ├── 01_DataBaseExtract.py       # Copies and preprocesses source data
│   ├── 02_TrainExecutive.py        # Organizes/prepares input and dispatches training
│   ├── 03_TrainCode.py             # Trains models and writes checkpoints
│   ├── 04_Evaluation.py            # Calculates evaluation statistics
│   ├── 05_Performance.py           # Produces predictions from saved checkpoints
│   └── Module/
│       ├── Config.py               # Paths, run times, regions and hyperparameters
│       ├── Model.py                # Encoder–decoder LSTM implementation
│       └── Utils.py                # Input processing, normalization and evaluation
├── Data/
│   ├── RawData/09h/                # Observation, forecast and derived inputs for SU
│   ├── ConcatData/data_09h.pickle  # Serialized prepared input dictionary
│   ├── 09h/PM2.5/SU/               # Season/day-specific Encoder, Decoder, Targets scaling files
│   ├── ForecasterEvaluation/       # Baseline-related evaluation files
│   └── ModelEvaluation/            # Model-related evaluation files
├── Result/
│   ├── WeightFile/09h/PM2.5/SU/    # TensorFlow checkpoint files
│   ├── TestResult/PM2.5/           # Prediction/observation/comparator tables
│   └── Evaluation/PM2.5/           # Evaluation tables
├── Read.txt                        # Brief original model description
├── README.md                       # EMS documentation, dataset and run guide
└── CHANGELOG_EMS.md                # Date-setting changes and limitations
```

The data and checkpoints cover **12 overlapping, three-month windows** (`JFM`, `FMA`, `MAM`, `AMJ`, `MJJ`, `JJA`, `JAS`, `ASO`, `SON`, `OND`, `NDJ`, `DJF`) and **three forecast-day groups** (`D0`, `D1`, `D2`). The archive contains 36 season/day pretrained checkpoint sets; each `save0` set contains `.meta`, `.index`, and `.data-00000-of-00001` files.

Although `Config.py` lists other regions, initialization hours and PM10, the supplied dataset and checkpoint hierarchy are primarily for the **SU / 09h / PM2.5** example. Additional configurations should not be described as publicly supported unless their files and execution are verified.

## 3. Data sources, file formats and variables

### 3.1. Provenance and availability

- **Data provider:** National Institute of Environmental Research (NIER), Republic of Korea (researcher-confirmed for all provided sources).
- **Input categories:** NIER-obtained observed air-quality/meteorological data, forecast variables (including the CMAQ comparison fields), and derived predictors stored under `anomaly`, `Cos` and `clcb` source groups.
- **Prepared input:** `Data/ConcatData/data_09h.pickle` (543,042,908 bytes; approximately **518 MiB**).
- **Scaling metadata:** Season/day-specific `Encoder.txt`, `Decoder.txt` and `Targets.txt` files hold feature names and scaling minimum/maximum values.
- **Text files:** Mainly tab-separated tables with `Date`, `Time` and variable fields; dates are coded `YYYYMMDD` in the supplied examples.
- **Availability:** The research team states training and validation datasets can be publicly shared. **[TO ADD]** Dataset deposit URL, release date, version, precise scope (source, processed, checkpoints, result files), licence and citation instructions. The archive is not yet claimed to be publicly deposited.

**Large-file handling:** The prepared pickle exceeds GitHub's standard 100 MB single-file limit. Deposit large files in an appropriate data repository, or provide a verified large-file solution, and link it explicitly from this README.

**Security:** Only deserialize pickle files from trusted sources; Python pickle loading can execute arbitrary code.

### 3.2. Input, target and output fields

Across the season/day scaling files, there are **12 distinct encoder variables, 21 distinct decoder variables and one target variable** (some configurations use subsets).

| Group | Fields in archive | Description and current unit status |
| --- | --- | --- |
| Observed pollution | `O_CO`, `O_NO2`, `O_O3`, `O_PM10`, `O_PM2_5`, `O_SO2` | Observed ambient pollutants. Researcher-confirmed examples: PM2.5 in **µg/m³**, NO₂ in **ppb**. Units of the other gases/PM10 and any hourly averaging conventions are **[TO CONFIRM]**. |
| Observed meteorology | `O_RH`, `O_RN_ACC`, `O_U`, `O_V`, `O_ta`, `O_td` | Meteorological inputs. The researcher reports temperature in **K**; confirm wind-component units/height, rainfall accumulation period, and remaining variable units. |
| Forecast pollution and surface meteorology | `f_CO`, `f_NO2`, `f_O3`, `f_PM10`, `f_PM2_5`, `f_RH`, `f_SO2`, `f_Ta` | Forecast predictors; `f_PM2_5` in **µg/m³**, `f_NO2` in **ppb**, `f_Ta` in **K**, assuming the stated units apply to both forecast and observed counterparts **[VERIFY AGAINST STORED DATA]**. Other exact units/lead-time conventions **[TO CONFIRM]**. |
| Forecast upper-air meteorology | `f_700hpa_V`, `f_850hpa_Ta`, `f_850hpa_gpm` | Forecast 700/850 hPa meteorological variables; 850 hPa temperature is reported in **K**; confirm wind units, geopotential height units and forecast conventions. |
| Cluster-related upper-air variables | `cs_700hpa_V`, `cs_850hpa_Ta`, `cs_850hpa_U`, `cs_850hpa_V`, `cs_850hpa_gpm` | **`cs_` is short for *cluster*** (researcher-confirmed). Precisely how each cluster-associated meteorological value is computed, assigned and encoded, and its physical units, remain **[TO CONFIRM]**. |
| Cluster-pattern predictors | `cluster_pattern_P01`–`cluster_pattern_P05` | Five atmospheric transport pattern classes described below. The stored representation (e.g., binary, probability, or other quantity) and clustering algorithm remain **[TO CONFIRM]**. |
| Target | `PM2_5` | PM2.5 concentration in **µg/m³**; confirm target time averaging and how it relates to `O_PM2_5`. |

#### Interpretation of the five cluster patterns (researcher-provided)

| Pattern | Atmospheric transport interpretation |
| --- | --- |
| `cluster_pattern_P01` | Stagnant airflow associated with the western coast / Yellow Sea coastal inflow. |
| `cluster_pattern_P02` | Short-range, direct inflow across the Yellow Sea. |
| `cluster_pattern_P03` | Airflow transported from Asian dust source regions. |
| `cluster_pattern_P04` | Transport from eastern China, between the Shandong and Shanghai areas. |
| `cluster_pattern_P05` | Northerly transport from the Manchurian region. |

**Unit convention:** The research team described units as SI-based; however, **µg/m³, ppb and K must each be specified explicitly**, and `ppb` is a mixing-ratio convention, not itself an SI unit. The values above are provided by the research team, not independently confirmed from unnormalized source files. Do not infer every pollutant or meteorological unit from these examples.

For publication and reuse, document variable definitions, physical units, observation height/spatial representation, averaging period, missing-value rules, and (for derived variables) generation formulas or software references. A `DATA_DICTIONARY.csv` is useful but not mandatory if these details are adequately included in the repository documentation.

Preprocessing recognizes `-999` and `-9999` in multiple code paths and sometimes replaces flags for precipitation/radiation with zero. The scientific justification of each variable-specific replacement is **[TO CONFIRM]**; do not treat missing pollutant concentrations as zero without evidence.

### 3.3. Six-hour time profile for the 09h experiment

The following mapping was supplied by the researchers in the six-hour forecasting time-profile table. `D−1` is the preceding day, `D0` the forecast date, `D+1` the following day and `D+2` two days later.

| Index | Relative day | Time interval in the supplied table |
| --- | --- | --- |
| T01 | D−1 | 07:00–12:00 |
| T02 | D−1 | 13:00–18:00 |
| T03 | D−1 | 19:00–24:00 |
| T04 | D0 | 01:00–06:00 |
| T05 | D0 | 07:00–12:00 |
| T06 | D0 | 13:00–18:00 |
| T07 | D0 | 19:00–24:00 |
| T08 | D+1 | 01:00–06:00 |
| T09 | D+1 | 07:00–12:00 |
| T10 | D+1 | 13:00–18:00 |
| T11 | D+1 | 19:00–24:00 |
| T12 | D+2 | 01:00–06:00 |
| T13 | D+2 | 07:00–12:00 |
| T14 | D+2 | 13:00–18:00 |
| T15 | D+2 | 19:00–24:00 |

For the **09h** configuration, the supplied table identifies **T01–T04 as observation-designated input periods** and T05–T15 as the forecast sequence. The source implements `T_OBS=4`, `T_LIST=5...15`, `D0` = T05–T07, `D1` = T08–T11, and `D2` = T12–T15. This is a **sequence classification**; it does not imply that every hour in T05 occurs strictly after the 09:00 issuance time.

**[TO CONFIRM]** Time zone (KST vs UTC), interpretation of interval endpoints, and whether all `Time` values in released tables follow this mapping exactly. The supplied original diagram also has 03h, 15h and 21h variants; the present archive specifically uses 09h.

## 4. Model methods and settings

Based on `Code/Module/Model.py`, `Code/Module/Config.py`, `Code/03_TrainCode.py` and `Read.txt`:

- **Architecture:** two-layer encoder–decoder LSTM with hidden dimensions `[96, 96]` and a fully connected output layer.
- **LSTM activation:** `tf.sin` in the legacy `tf.contrib.rnn.LSTMCell` implementation.
- **Optimization:** root mean squared error (RMSE) loss; Adam optimizer with global-norm gradient clipping of `1`.
- **Batch size:** `64`.
- **Learning rates:** up to five scheduled stages starting at `1e-4`, decreasing by a factor of 10 for each subsequent stage.
- **Random seeds:** `42` for Python, NumPy and TensorFlow where defined; these do not guarantee bitwise reproducibility across platforms.
- **Feature screening:** original documentation gives a Pearson-correlation threshold of `|R| >= 0.3` and `p <= 0.01`; refer to `InputSet` in `Module/Utils.py` for the actual selection procedure.
- **Normalization:** per-model variable/time-step Min–Max parameters retained as `Encoder.txt`, `Decoder.txt`, and `Targets.txt`.

## 5. Experiment periods and revised training scripts

**Researcher-confirmed experiment definition:** T5V21 uses **training data from 2016-01-01 through 2020-12-31 (five years)** and **validation data from 2021-01-01 through 2021-12-31**. This agrees with the original `Read.txt`.

To align the distributed training scripts with the stated experiment, **this revised package changes the following settings**:

| Script | Parameter | Revised value |
| --- | --- | --- |
| `Code/02_TrainExecutive.py` | Date range for training-period feature screening (`train_date`) | `20160101`–`20201231` |
| `Code/03_TrainCode.py` | Training-data selection (`train_date`) | `20160101`–`20201231` |
| `Code/03_TrainCode.py` | Held-out validation-data selection (`test_date` in the unchanged `Read` API) | `20210101`–`20211231` |

In the **originally uploaded archive**, the training script instead selected data from 2016–2021 and used 2022 as a held-out period. Those conflicting values have been corrected **only in the distributed Python source**. The existing functions still name the held-out split `test_date` or `test`, but for this experiment that split is used as the **2021 validation set**.

**Important reproducibility limitation:** The prepared `Data/ConcatData/data_09h.pickle`, season-specific feature lists/scaling metadata and existing `Result/WeightFile/` checkpoints were copied unchanged. Their creation dates, feature-selection period and training provenance cannot be established solely by editing source code. Re-running training with the corrected years can produce different weights and results. Before claiming exact reproduction of manuscript metrics, verify whether the archived artifacts were produced under T5V21 or regenerate the necessary artifacts using the corrected dates.

**Prediction/evaluation periods are independent:** `Read.txt` describes a 2023-01-01–2023-03-31 prediction example. The supplied `05_Performance.py` remains configured for 2024-01-01–2024-03-31 (`target_year = 2024`), and `04_Evaluation.py` remains configured for `EVAL_YEAR = 2024`; these were **not changed**, because they are forecast-output settings rather than the model's train/validation split. If the paper reports a different independent forecast period, select that period and verify the prepared data and reference tables. In the optional data-preparation stage, `02_TrainExecutive.py` has a separate overall input date range ending on 2024-03-30; it is not the training cutoff.

## 6. Software environment and system requirements

### 6.1. Current NIER login-node information (observed, not necessarily used for training)

The information below comes from environment reports supplied by the researchers. It describes the **currently inspected login node**, not a confirmed original training environment and not minimum system requirements.

| Item | Observed information |
| --- | --- |
| Operating system | Rocky Linux 8.10 |
| CPU | Intel Xeon Gold 6346 at 3.10 GHz |
| Logical CPU count | 64 |
| System memory | Approximately 251 GiB |
| Default system Python | Python 3.6.8 at `/usr/bin/python3` |
| Default-environment TensorFlow | Not found |
| Available site modules (examples) | Python 3.7.6 with `tensor23` in the name; Python 3.8.10 with `tensor212` in the name |
| GPU/CUDA on current login session | `nvidia-smi` and `nvcc` commands not found; this is **not evidence** that the training node lacked a GPU |

### 6.2. Original model-development environment (researcher-reported)

| Item | Reported value | Confidence |
| --- | --- | --- |
| Python | **3.5.2** | Reported by research team |
| TensorFlow | **1.8.0** | Reported by research team |
| CUDA toolkit | **9.0.176** | Reported by research team |
| Operating system of original training node | **[TO CONFIRM]** | Current Rocky Linux node cannot be assumed to match |
| GPU model and GPU memory | **[TO CONFIRM]** | `DeviceNumber = "3"` names an indexed device, not a GPU model |
| cuDNN | **[TO CONFIRM]** | Needed if GPU TensorFlow was used |
| NumPy, pandas, SciPy, scikit-learn versions | **[TO CONFIRM]** | Do not infer from today's login node |

The source uses TensorFlow 1.x APIs such as `tf.Session`, `tf.placeholder` and `tf.contrib.rnn`. TensorFlow 2.x is not a validated drop-in replacement. The updated distribution has **not** been re-executed with this reported historical toolchain. A version-pinned and tested environment description should be provided at release; unsupported dependency versions must not be invented.

## 7. Running the supplied code (researcher-reported; revised package untested)

**Prerequisites before trying the following commands:** activate a compatible historical Python 3.5.2/TensorFlow 1.8.0 environment (and CUDA 9.0.176 if using the original GPU configuration), obtain the large prepared dataset if hosted separately, and address any institution-specific paths in a separate working copy. The revised date settings have not been independently executed.

1. In `Code/Module/Config.py`, replace the original institution-specific `ABS_PATH` and any other required source paths with accessible project paths. Some code concatenates directory strings, so the trailing separator matters.
2. In `Code/Module/Utils.py`, resolve the original absolute module search path and `from Config import ...` statement to a portable module import.
3. Update or parameterize `DeviceNumber = "3"` and the GPU selection in `05_Performance.py` for the available system.
4. Confirm the 36 model checkpoints and corresponding scaling files are accessible. The prepared `data_09h.pickle` is necessary for the distributed prediction script.
5. Avoid directly running the institutional database-copy script unless the external database/export paths exist and permission to access them is documented.

Once these prerequisites are met, the intended working directory and **code-derived entry points** are:

```bash
cd T5V21/Code

# Predict using saved model weights; review target year and output paths first.
python 05_Performance.py

# Evaluate prediction tables; review EVAL_YEAR and input file location first.
python 04_Evaluation.py

# Full retraining is an additional step and has not been verified:
# python 02_TrainExecutive.py
```

**Cautions:**
- `05_Performance.py` currently writes `Result/TestResult/PM2.5/SU_09_2024.txt` and may remove an existing file at that path. Use a working copy or back up outputs.
- The archive instead stores a supplied 2024 output at `Result/TestResult/PM2.5/T5V21/SU_09_2024.txt`; `04_Evaluation.py` expects the non-`T5V21/` path. Align them intentionally and record the adjustment.
- `02_TrainExecutive.py` currently has `make_data = False` and `Train = True`. Its training code uses the wider `SMA` regional group in places, even though the ZIP's detailed trained-model hierarchy is limited to `SU`. The available archive has **not** been demonstrated to contain all inputs needed for end-to-end retraining.
- Raw-data preprocessing may overwrite files. Work from copies, and do not run the database extraction script indiscriminately.

## 8. Evaluation products and verification

The distributed prediction tables use the columns `Date`, `Time`, `Target`, `AI` and `CMAQ`:
- `Target`: observed target concentration.
- `AI`: prediction from the LSTM-based model.
- `CMAQ`: comparison-model forecast.

`04_Evaluation.py` computes numerical statistics such as RMSE, correlation coefficient (`R`), index of agreement (`IOA`) and normalized mean bias (`NMB`), and event/category scores including `ACC`, `POD`, `FAR` and `F1`. For precise definitions and thresholds, consult the functions in `Module/Utils.py` and match the evaluation years/temporal aggregations used in the manuscript.

**Reproducibility status:** The researchers report that the original code runs successfully, generates PM2.5 predictions and supports performance evaluation. The revised source and archived checkpoints have **not** been independently re-tested or verified in a clean environment. A source-date correction alone does not prove that supplied historical weights or tabulated metrics were produced with a 2016–2020/2021 split. Release a reproducible example with reference results and a stated numerical tolerance if possible.

## 9. Software and data availability for release

To make the public deposit consistent with the FAIR principles (Findable, Accessible, Interoperable and Reusable):

- **Code:** Publish the approved Python source, this README, a verified dependency specification and an explicit licence through a stable public repository. Supply the exact URL and release identifier.
- **Data:** Deposit the permitted input data, normalization metadata and reference results with variable definitions, file formats, licensing, version information and a persistent URL/DOI where available. Cite the data source as NIER, with original production-system provenance where relevant.
- **Model artifacts:** State whether pretrained checkpoint sets are included and provide their release location. **[TO CONFIRM]** Publication permission for each class of derived data and weights.
- **Hardware/software requirements:** Replace unknowns with the actual *tested* environment and distinguish it from the current login-node inventory.
- **Cost and access restrictions:** State the true software and data access terms at the date of public release; **do not write “contact the authors” as the only access route**.
- **Security and cleanliness:** Exclude secrets, credentials, unnecessary `__pycache__/` directories and unreviewed internal server paths from the public deposit.

### Draft manuscript subsection: Software and Data Availability

*Complete the remaining placeholders and insert a bulleted **Software and Data Availability** section after the manuscript's Highlights, as required in the editor's letter. Do not confuse a draft availability claim with a currently accessible public deposit.*

- **Name of software:** T5V21, LSTM-based PM2.5 forecasting model and experimental code.
- **Developers:** KIM SEOYOEN, National Institute of Environmental Research (NIER), Republic of Korea.
- **Contact:** KIM SEOYOEN, `ksy0720@korea.kr`.
- **Date first available:** **[TO ADD first public release date]**.
- **Programming language:** Python 3.5.2 (original development environment).
- **Software required:** TensorFlow 1.8.0, CUDA 9.0.176 (GPU use), NumPy, pandas, SciPy, scikit-learn (**[TO ADD actual package versions/cuDNN if relevant]**).
- **Hardware required:** **[TO ADD verified original/tested hardware or minimum requirements]**.
- **Program size:** Approximately 69.3 KiB Python source code in the original package (**[UPDATE final release size]**).
- **Source code and documentation:** **[TO ADD publicly accessible GitHub/repository URL, release tag and README link]**.
- **Cost and licence:** **[TO ADD confirmed use terms and NIER-approved licence]**.
- **Data source:** NIER observation, numerical forecast and cluster-related environmental inputs. Training: 2016–2020; validation: 2021.
- **Data description:** Observed/forecast PM2.5 (µg/m³); NO₂ (ppb); temperature (K); derived cluster-related fields and five atmospheric transport patterns. Confirm other field units and cluster-generation metadata before release.
- **Data repository/size:** Prepared `data_09h.pickle` approximately 518 MiB; **[TO ADD data URL/DOI, release version, access terms and licence]**.
- **Pretrained checkpoints and reference outputs:** **[TO ADD deposit location and reuse conditions]**.

## 10. Remaining author confirmations before release

1. **Check the provenance of distributed artifacts:** Confirm whether the saved checkpoints, feature-selection lists, prepared pickle and manuscript statistics belong to the 2016–2020 training / 2021 validation experiment; regenerate as needed. The Python source dates were corrected, but outputs were not regenerated.
2. **Complete the developer and contact record:** `KIM SEOYOEN` and `ksy0720@korea.kr` were supplied; verify preferred case/romanization, author contribution statement and any additional developers if applicable.
3. **Complete remaining variable metadata:** Other pollutant and meteorological units; `cs_` numerical construction; exact encoding and derivation of `cluster_pattern_P01`–`P05`; time zone, averages and missing-value processing.
4. **Complete environment metadata:** Original training OS, GPU model, cuDNN, pinned NumPy/pandas/SciPy/scikit-learn versions; demonstrate a supported installation if possible.
5. **Set public release conditions:** NIER-approved software/data licences, precise redistribution scope, public GitHub/data links, original public release date, access/cost and reference results.

**Manuscript consistency note:** The manuscript's data/software availability subsection, model methods, experiment dates and any released repository documentation must agree. The editor also requested Highlights, a CRediT statement, removal of manuscript line numbers and a revised cover letter; those are separate manuscript/submission actions rather than README content.
