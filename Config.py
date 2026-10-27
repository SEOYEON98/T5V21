#
# ==============================================================================
#   Program Name   : Config
# ------------------------------------------------------------------------------
#   Description
# ------------------------------------------------------------------------------
#   Company Name   : NIER (National Institute of Environmental Research)
#   Create Date    : 2022-04-13
# ------------------------------------------------------------------------------
#   Update History : 2024-06-01
# ==============================================================================
#

# --------------------
# Set library.
#
import os

# --------------------
# Set path.
#
ABS_PATH = '/DATA/AI-modeling/HSC/T5V21/'
NIER_DB_DATA_PATH = "/DATA/AI-modeling/Anyang/AIFM/interface/nier/LearningDataRnn1_7/{{RUN_TIME}}h/{{TYPE}}/"
AU_DB_DATA_PATH = "/DATA/AI-modeling/Anyang/AIFM/interface/nier/LearningDataDnn1/Backup/ANYANG_DATA/{{RUN_TIME}}h/{{TYPE}}/"
RAW_DATA_PATH = os.path.join(ABS_PATH, 'Data/RawData/{{RunTime}}h/')
MINMAX_PATH = os.path.join(ABS_PATH, 'Data/{{RunTime}}h/{{PM}}/{{Area}}/{{Season}}/{{Day}}/')
LOG_PATH = os.path.join(ABS_PATH, 'Result/Log/{{RunTime}}h/{{PM}}/')
SAVE_PATH = os.path.join(ABS_PATH, 'Result/WeightFile/{{RunTime}}h/{{PM}}/{{Area}}/{{Season}}/{{Day}}/')

# --------------------
# Set common parameters.
#
# Reference :: season & month #
season_box = ['JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF']
month_box = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [11, 12, 1], [12, 1, 2]]
season_month = {'JFM': [1, 2, 3], 'FMA': [2, 3, 4], 'MAM': [3, 4, 5], 'AMJ': [4, 5, 6],
                'MJJ': [5, 6, 7], 'JJA': [6, 7, 8], 'JAS': [7, 8, 9], 'ASO': [8, 9, 10],
                'SON': [9, 10, 11], 'OND': [10, 11, 12], 'NDJ': [11, 12, 1], 'DJF': [12, 1, 2]}

# Reference :: Data type #
DATA_TYPE = {'OBSERVATION'   :{'FOLDER_NAME':'obs'     , 'FILE_NAME':'obs'},
             'FORECAST'      :{'FOLDER_NAME':'forecast', 'FILE_NAME':'f'},
             'ANOMALY'       :{'FOLDER_NAME':'anomaly' , 'FILE_NAME':'anm'},
             'COSIN'         :{'FOLDER_NAME':'Cos'     , 'FILE_NAME':'cos'},
             'CONTRI&CLUSTER':{'FOLDER_NAME':'clcb'    , 'FILE_NAME':'clcb'}}

# Reference :: PM #
PMS = ['PM2_5', 'PM10']
PM_DIR = {'PM2.5': {'PM_NAME': 'PM2_5', 'PM_OBS': 'O_PM2_5', 'PM_FORECAST': 'f_PM2_5'},
          'PM10' : {'PM_NAME': 'PM10' , 'PM_OBS': 'O_PM10' , 'PM_FORECAST': 'f_PM10' }}

# Reference :: Model run time #
MODEL_TIMES = {'03': {'T_LIST': [i for i in range(4, 16)], 'T_OBS': 3, 'D0_LEN': 4},
               '09': {'T_LIST': [i for i in range(5, 16)], 'T_OBS': 4, 'D0_LEN': 3},
               '15': {'T_LIST': [i for i in range(6, 16)], 'T_OBS': 5, 'D0_LEN': 2},
               '21': {'T_LIST': [i for i in range(7, 16)], 'T_OBS': 6, 'D0_LEN': 1}}

# Reference :: Area #
AREAS = ['SU', 'IC', 'GGN', 'GGS', 'GWW', 'BS', 'DG', 'GSS', 'GSN', 'US', 'DJ', 'SJ', 'CCN', 'CCS', 'JLS', 'JLN', 'GJ', 'GWE', 'JJ']
AREA_GROUP = {'SMA': ['SU', 'IC', 'GGN', 'GGS', 'GWW'],
              'CC' : ['DJ', 'SJ', 'CCN', 'CCS']       ,
              'HN' : ['GJ', 'JLN', 'JLS']             ,
              'YN' : ['BS', 'DG', 'US', 'GSN', 'GSS'] ,
              'GWE': ['GWE']                          ,
              'JJ' : ['JJ']                           }

# Reference :: Drop cloumns #
Drop = ['PM10_', 'PM2.5_', 'hpa_W', 'f_gpm', 'f_RN_ACC', 'f_MH', 'O_radiation', 'f_temp_850_925']

# --------------------
# Set train parameters.
#
counts, batch_size, state_sizes = 5, 64, [96, 96]
DeviceNumber = "3"
