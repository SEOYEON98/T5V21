#
# ==============================================================================
#   Program Name   : TrainExecutive
# ------------------------------------------------------------------------------
#   Description
# ------------------------------------------------------------------------------
#   Company Name   : NIER (National Institute of Environmental Research)
#   Create Date    : 2022-04-13
# ------------------------------------------------------------------------------
#   Update History : 2022-06-10
# ==============================================================================
#

# --------------------
# Set library.
#
import os
import sys
import subprocess
import pickle
import numpy as np
import datetime as d
from Module.Config import ABS_PATH, RAW_DATA_PATH, LOG_PATH, PMS, AREAS, MODEL_TIMES, Drop, month_box, season_box
from Module.Utils import TotalDate, ConcatData, InputSet, Directory


# --------------------
# Argument.
#
make_data = False
Train = True


# --------------------
# Run code.
#
if make_data:
    DICT_PATH = os.path.join(ABS_PATH, 'Data', 'ConcatData') + os.sep
    Directory(path=DICT_PATH, remove=False, make=True)
    for PM in PMS[0:1]:

        for RT in ['09']:
            separation, day_len = MODEL_TIMES[RT]['T_OBS'], MODEL_TIMES[RT]['D0_LEN']
            data_path = RAW_DATA_PATH.replace('{{RunTime}}', RT)
            _, total_date = TotalDate(start_date='20160101', end_date='20240330')
            _, train_date = TotalDate(start_date='20160101', end_date='20201231')
            data_dict, save_dict = {}, DICT_PATH + 'data_{}h.pickle'

            for A in AREAS:
                data_dict[A+'_name'], data_dict[A+'_date'], data_dict[A+'_data'] = ConcatData(area=A, path=data_path, total_length=total_date, drop_col=Drop)
                print("Concat data :: PM={} / RunTime={} / Area={}".format(PM, RT, A))
                for season, month in zip(season_box, month_box):
                    data_dict[A+'_{}_use_col'.format(season)] = InputSet(dict=data_dict, region=A, month=month, PM_name=PM, train_date=train_date)
                    print("Input variables selection :: PM={} / RunTime={} / Area={} / Season={} / Month={}".format(PM, RT, A, season, month))
                print("\n")
            print("\n")

        # Delete and Save #
        if os.path.isfile(save_dict.format(RT)): os.remove(save_dict.format(RT))
        with open(save_dict.format(RT), 'wb') as fw: pickle.dump(data_dict, fw)


# --------------------
# Generate arguments dictionary.
#
if Train:
    key_name = ['PM', 'MODEL_TIME', 'AREA', 'SEASON', 'DAY_LENGTH']
    ARG_DICT = []
    for PM in PMS[0:1]:
        if PM == 'PM2_5': pm_name = 'PM2.5'
        if PM == 'PM10' : pm_name = 'PM10'

        for RT in sorted(MODEL_TIMES.keys())[1:2]:
            LogPath = LOG_PATH.replace('{{RunTime}}', RT).replace('{{PM}}', pm_name)
            Directory(path=LogPath, remove=False, make=True)

#            for A in ['SU','IC','GGN','GGS','GWW','BS','DG','GSS','GSN','US','DJ','SJ','CCN','CCS','JLS','JLN','GJ','GWE','JJ']:
            for A in ['SU']:
                LogFile = open(LogPath + '{}_TrainLog.txt'.format(A), 'w')

#                for season in season_box[0:1]:
                for season in season_box:

                    for DAY_N in ['D0', 'D1', 'D2']:
                        print("Train parameter ::: PM={} / ModelTime={} / Area={} / Season={} / TargetDay={}".format(PM, RT, A, season, DAY_N))
                        key_value = [PM, RT, A, season, DAY_N]
                        ARG_DICT.append({at: k for at, k in zip(key_name, key_value)})
                        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), '03_TrainCode.py'), PM, RT, A, season, DAY_N])
                        print("\n")
                        print("\n", file=LogFile)
