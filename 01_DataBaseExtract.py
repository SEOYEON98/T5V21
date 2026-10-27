#
# ==============================================================================
#   Program Name   : DataBaseExtract
# ------------------------------------------------------------------------------
#   Description
# ------------------------------------------------------------------------------
#   Company Name   : NIER (National Institute of Environmental Research)
#   Create Date    : 2022-04-13
# ------------------------------------------------------------------------------
#   Update History : 2022-06-10
# ==============================================================================
#

# ---------------------------------
# Set library.
#
import os
import sys
# sys.path.append("/DATA/AI-modeling/JBLee/Development/AI/LSTM/FASAPv1.0.0/Code/")
from Module.Utils import DB_COPY, ANYANG_DB_COPY, FORECAST_VALUE_CHANGE, FORECAST03_CHANGE, OBS_VALUE_CHANGE
from Module.Config import ABS_PATH, NIER_DB_DATA_PATH, AU_DB_DATA_PATH, MODEL_TIMES, AREAS, DATA_TYPE

# ---------------------------------
# Set path.
#
NIER_OUT_PATH = os.path.join(ABS_PATH, "Data/RawData/{{RUN_TIME}}h/{{TYPE}}/")

# ---------------------------------
# DB data preprocessing.
#
for RT in sorted(MODEL_TIMES.keys())[1:2]:

    for A in AREAS:

        DB_COPY(IN_PATH=NIER_DB_DATA_PATH, OUT_PATH=NIER_OUT_PATH, RUN_TIME=RT, AREA=A, TYPE=DATA_TYPE.keys())
        print("========== Finish DB data copy !!! : RUN_TIME={}h / AREA={} ==========".format(RT, A))


        FORECAST_VALUE_CHANGE(IN_PATH=NIER_OUT_PATH, RUN_TIME=RT, AREA=A)
        print("========== Finish Forecast value change !!! : RUN_TIME={}h / AREA={} ==========".format(RT, A))

        '''
        if RT == '09':
            FORECAST03_CHANGE(IN_PATH_03=NIER_OUT_PATH, IN_PATH_09=NIER_OUT_PATH, RUN_TIME=RT, AREA=A)
            print("========== Finish Forecast 03h value change !!! : RUN_TIME={}h / AREA={} ==========".format(RT, A))
        '''

        OBS_VALUE_CHANGE(IN_PATH=NIER_OUT_PATH, AU_IN_PATH=AU_DB_DATA_PATH, RUN_TIME=RT, AREA=A)
        print("========== Finish obs value change !!! : RUN_TIME={}h / AREA={} ==========".format(RT, A))
        print("\n")
