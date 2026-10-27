# EMS submission preparation

- Training period in `Code/02_TrainExecutive.py` and `Code/03_TrainCode.py` changed to 2016-01-01 through 2020-12-31.
- Held-out validation period in `Code/03_TrainCode.py` changed to 2021-01-01 through 2021-12-31 (variable name `test_date` retained).
- Original model weights, preprocessed datasets, and evaluation results were **not** re-created or altered.
- Prediction/evaluation settings targeting 2024 were not changed.
- This is **source code and documentation only**. Large data files, pretrained checkpoints, and results must be deposited separately before claiming fully reproducible public software.
- Before public upload, inspect original institutional absolute paths in `Code/Module/Config.py`, `Code/Module/Utils.py`, and scripts for information that should not be made public.
- The corrected source package has passed a syntax parse check only; the modified scripts have **not** been executed.
