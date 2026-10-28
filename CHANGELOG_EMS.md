# EMS submission preparation

- Training period in `Code/02_TrainExecutive.py` and `Code/03_TrainCode.py` changed to 2016-01-01 through 2020-12-31.
- Held-out validation period in `Code/03_TrainCode.py` changed to 2021-01-01 through 2021-12-31 (variable name `test_date` retained).
- Original model weights, preprocessed datasets, and evaluation results were **not** re-created or altered.
- Prediction/evaluation settings targeting 2024 were not changed.
- This is **source code and documentation only**. Large data files, pretrained checkpoints, and results must be deposited separately before claiming fully reproducible public software.
- Before public upload, inspect original institutional absolute paths in `Code/Module/Config.py`, `Code/Module/Utils.py`, and scripts for information that should not be made public.
- The corrected source package has passed a syntax parse check only; the modified scripts have **not** been executed.

## Portable paths and optional GPU configuration (2026-10-28)

- `Code/Module/Config.py`: path templates are rooted in the extracted repository by default; `T5V21_ROOT` may override the root. Optional source-data locations use `T5V21_NIER_SOURCE` and `T5V21_ANYANG_SOURCE`.
- `Code/Module/Utils.py`: removed the institute-specific `sys.path` insertion; imports now resolve through `Module.Config`.
- `Code/03_TrainCode.py` and `Code/05_Performance.py`: GPU pinning to device 3 was removed; set `T5V21_CUDA_DEVICES` if needed, before TensorFlow is imported.
- `Code/02_TrainExecutive.py`: training subprocess uses the current Python interpreter and script directory instead of `os.system('python ...')`.
- `Code/01_DataBaseExtract.py`, `Code/02_TrainExecutive.py`, `Code/03_TrainCode.py`, `Code/04_Evaluation.py`, and `Code/05_Performance.py`: standardized local Data/ and Result/ paths.
- **No model equations, training period, validation period, checkpoints, or input data changed in this portability update.**
- Syntax/import/path-resolution checks were performed. A complete TensorFlow 1.8 model training or inference test was not performed.
