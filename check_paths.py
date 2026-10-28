# -*- coding: utf-8 -*-
"""Read-only check of the expected extracted T5V21 data/result folders."""
from __future__ import print_function
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, 'Code'))
from Module.Config import ABS_PATH

checks = [
    ('Prepared input data', 'Data/ConcatData/data_09h.pickle', 'file'),
    ('Scaling metadata', 'Data/09h/PM2.5/SU', 'dir'),
    ('Saved LSTM checkpoints', 'Result/WeightFile/09h/PM2.5/SU', 'dir'),
    ('Prediction output folder (optional)', 'Result/TestResult/PM2.5', 'dir'),
]

print('T5V21 root: {}'.format(ABS_PATH))
missing_required = 0
for idx, (label, relative_path, kind) in enumerate(checks):
    p = os.path.join(ABS_PATH, *relative_path.split('/'))
    exists = os.path.isfile(p) if kind == 'file' else os.path.isdir(p)
    print('[{}] {}: {}'.format('OK' if exists else 'MISSING', label, p))
    if not exists and idx < 3:
        missing_required += 1

if missing_required:
    print('Required materials missing. Extract the GitHub data and result release assets so their Data/ and Result/ directories are under T5V21_ROOT.')
    sys.exit(1)
print('Required locations exist. This check does not verify file content or model execution.')
