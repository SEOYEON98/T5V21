#
# ==============================================================================
#   Program Name   : Utils
# ------------------------------------------------------------------------------
#   Description
# ------------------------------------------------------------------------------
#   Company Name   : NIER (National Institute of Environmental Research)
#   Create Date    : 2022-04-13
# ------------------------------------------------------------------------------
#   Update History : 2024-04-28
# ==============================================================================
#

# -*- coding: utf-8 -*-
import os
import math
import shutil
import calendar
import numpy as np
import pandas as pd
import datetime as d
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# I don't want to print warning sign causing due to nan #
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from Module.Config import MODEL_TIMES, DATA_TYPE


# --------------------
# Calculate of total date.
#
def TotalDate(start_date, end_date):
    total_date, custom_date = [], []
    s_year, s_month, s_day = int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8])
    e_year, e_month, e_day = int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8])
    FromDateTime = d.datetime.strptime('{}-{}-{}'.format(s_year, s_month, s_day), "%Y-%m-%d")
    ToDateTime = d.datetime.strptime('{}-{}-{}'.format(e_year, e_month, e_day), "%Y-%m-%d")
    NowDateTime = FromDateTime

    while NowDateTime <= ToDateTime:
        year = str(NowDateTime.year).zfill(4)
        month = str(NowDateTime.month).zfill(2)
        day = str(NowDateTime.day).zfill(2)
        format_date = '{}{}{}'.format(year, month, day)
        custom_date.append(int(format_date))
        total_date.append(NowDateTime)

        NowDateTime = NowDateTime + d.timedelta(days=1)

    return total_date, custom_date


# ---------------------------------
# NIER DB data copy. (obs, forecast, clcb, cos, anomaly)
#
def DB_COPY(IN_PATH, OUT_PATH, RUN_TIME, AREA, TYPE):
    for DT in TYPE:
        folder_name = DATA_TYPE[DT]['FOLDER_NAME']
        file_name = DATA_TYPE[DT]['FILE_NAME']

        DATA_PATH = IN_PATH.replace("{{RUN_TIME}}", RUN_TIME).replace("{{TYPE}}", folder_name)
        RESULT_PATH = OUT_PATH.replace("{{RUN_TIME}}", RUN_TIME).replace("{{TYPE}}", folder_name)
        if not os.path.isdir(RESULT_PATH): os.makedirs(RESULT_PATH)

        try:
            shutil.copy(os.path.join(DATA_PATH + "{}_{}_{}.txt".format(AREA, file_name, RUN_TIME)), RESULT_PATH)
        except OSError:
            print("DB_COPY ::: Not found : Run_time={}h / Data_type={}".format(RUN_TIME, DT))


# ---------------------------------
# Observation data of Anyang DB is copied.
#
def ANYANG_DB_COPY(IN_PATH, OUT_PATH, RUN_TIME, AREA, TYPE):
    for DT in TYPE:
        if (DT == 'FORECAST') or (DT == 'ANOMALY') or (DT == 'COSIN') or (DT == 'CONTRI&CLUSTER'):
            pass

        elif DT == 'OBSERVATION':
            folder_name = DATA_TYPE[DT]['FOLDER_NAME']
            file_name = DATA_TYPE[DT]['FILE_NAME']
            DATA_PATH = IN_PATH.replace("{{RUN_TIME}}", RUN_TIME).replace("{{TYPE}}", folder_name)
            RESULT_PATH = OUT_PATH.replace("{{RUN_TIME}}", RUN_TIME).replace("{{TYPE}}", folder_name)
            if not os.path.isdir(RESULT_PATH): os.makedirs(RESULT_PATH)

            try:
                shutil.copy(os.path.join(DATA_PATH + "{}_{}_{}.txt".format(AREA, file_name, RUN_TIME)), RESULT_PATH)
            except OSError:
                print("ANYANG_DB_COPY ::: Not found : Run_time={}h / Data_type={}".format(RUN_TIME, DT))


# ---------------------------------
# -999.0 or -9999.0 value of 'f_RN_ACC', One of the forecast data, is changed to 0.
#
def FORECAST_VALUE_CHANGE(IN_PATH, RUN_TIME, AREA):
    NIER_RAW = IN_PATH.replace("{{RUN_TIME}}", RUN_TIME).replace("{{TYPE}}", "forecast")

    NIER_DATA = pd.read_csv(NIER_RAW + "{}_f_{}.txt".format(AREA, RUN_TIME), delimiter="\t")
    NIER_LIST = NIER_DATA.loc[(NIER_DATA.Time >= MODEL_TIMES[RUN_TIME]['T_LIST'][0])]
    NIER_DATE = NIER_LIST['Date'].tolist()

    NIER_DATA['f_RN_ACC'] = NIER_DATA['f_RN_ACC'].mask(NIER_DATA['f_RN_ACC'] == -9999, 0)
    NIER_DATA['f_RN_ACC'] = NIER_DATA['f_RN_ACC'].mask(NIER_DATA['f_RN_ACC'] == -999, 0)
    NIER_DATA.to_csv(NIER_RAW + "{}_f_{}.txt".format(AREA, RUN_TIME), sep="\t", index=False)


# ---------------------------------
# From T12 to T15 of 03h is replaced from T12 to T15 of 09h.
#
def FORECAST03_CHANGE(IN_PATH_03, IN_PATH_09, RUN_TIME, AREA):

    if (RUN_TIME == '03') or (RUN_TIME == '15') or (RUN_TIME == '21'):
        pass

    elif (RUN_TIME == '09'):
        THR_path = IN_PATH_03.replace("{{RUN_TIME}}", '03').replace("{{TYPE}}", "forecast")
        FOUR_path = IN_PATH_09.replace("{{RUN_TIME}}", '09').replace("{{TYPE}}", "forecast")

        THR_DATA = pd.read_csv(THR_path + "{}_f_03.txt".format(AREA), delimiter="\t")
        FOUR_DATA = pd.read_csv(FOUR_path + "{}_f_09.txt".format(AREA), delimiter="\t")

        FOUR_F_DATA = FOUR_DATA.loc[(FOUR_DATA['Date'] < 20170100) & (FOUR_DATA['Time'] >= 12)]
        DROP_INDEX = FOUR_F_DATA.index
        THR_F_DATA_DROP = THR_DATA.drop(DROP_INDEX)

        THR_F_DATA_DROP = THR_F_DATA_DROP.append(FOUR_F_DATA).reset_index(drop=True)
        THR_F_DATA_DROP = THR_F_DATA_DROP.sort_values(by=['Date', 'Time']).reset_index(drop=True)

        THR_F_DATA_DROP['f_RN_ACC'] = THR_F_DATA_DROP['f_RN_ACC'].mask(THR_F_DATA_DROP['f_RN_ACC'] == -9999, 0)
        THR_F_DATA_DROP['f_RN_ACC'] = THR_F_DATA_DROP['f_RN_ACC'].mask(THR_F_DATA_DROP['f_RN_ACC'] == -999, 0)
        THR_F_DATA_DROP.to_csv(THR_path + "{}_f_03.txt".format(AREA), sep="\t", index=False)


# ---------------------------------
# -999.0 or -9999.0 value of 'O_RN_ACC' and 'O_radiation', among the observation data, is changed to 0.
# From 2016.01.01 to 2016.06.30 of NIER observation data is replaced from Anyange observation data.
#
def OBS_VALUE_CHANGE(IN_PATH, AU_IN_PATH, RUN_TIME, AREA):
    OBS_path = IN_PATH.replace("{{RUN_TIME}}", RUN_TIME).replace("{{TYPE}}", "obs")
    AU_OBS_path = AU_IN_PATH.replace("{{RUN_TIME}}", RUN_TIME).replace("{{TYPE}}", "obs")

    # If, 'O_RN_ACC' values is -9999, replace 0 #
    OBS_DATA = pd.read_csv(OBS_path + "{}_obs_{}.txt".format(AREA, RUN_TIME), delimiter="\t")
    OBS_DATA['O_RN_ACC'] = OBS_DATA['O_RN_ACC'].mask((OBS_DATA['O_RN_ACC'] == -9999) | (OBS_DATA['O_RN_ACC'] == -999), 0)
    OBS_DATA['O_radiation'] = OBS_DATA['O_radiation'].mask((OBS_DATA['O_radiation'] == -9999) | (OBS_DATA['O_RN_ACC'] == -999), 0)
    OBS_DATA = OBS_DATA.loc[(OBS_DATA.Date > 20160600)]

    AU_OBS_DATA = pd.read_csv(AU_OBS_path + "{}_obs_{}.txt".format(AREA, RUN_TIME), delimiter="\t")
    AU_OBS_DATA = AU_OBS_DATA.loc[(AU_OBS_DATA.Date < 20160600)]
    ta_list, td_list = [], []
    ta = AU_OBS_DATA['O_ta'].tolist(); td = AU_OBS_DATA['O_td'].tolist()
    for i in range(len(ta)):
        if ta[i] == -9999 or ta[i] == -999:
            ta_list.append(ta[i])
        else:
            nier_ta = ta[i] - 273
            ta_list.append(nier_ta)

    for i in range(len(td)):
        if td[i] == -9999 or td[i] == -999:
            td_list.append(td[i])
        else:
            nier_td = td[i] - 273
            td_list.append(nier_td)

    AU_OBS_DATA['O_ta'] = ta_list
    AU_OBS_DATA['O_td'] = td_list

    MERGE_OBS = pd.concat([AU_OBS_DATA, OBS_DATA])
    MERGE_OBS.to_csv(OBS_path + "{}_obs_{}.txt".format(AREA, RUN_TIME), sep="\t", index=False)


# --------------------
# All raw data is concatenated from dictionary type.
#
def ConcatData(area, path, total_length, drop_col):
    # List of data type #
    data_kind = os.listdir(path)

    # Empty lists are filled from text #
    txt_box = []
    for d in data_kind:
        if d == 'anomaly':
            continue
        data_name = os.listdir(path + d)
        for data in data_name:
            if area in data:
                txt_box.append(path + d + '/' + data)

    # Make concatenate data #
    NAME, VAR = np.empty(0, dtype=np.str), np.empty((len(total_length), 15, 0), dtype=np.float)
    for txt in txt_box:
        var_name = np.genfromtxt(txt, dtype=np.str, skip_header=0)[0][2:]
        var_value = np.genfromtxt(txt, dtype=np.float, skip_header=1)[:, 2:]
        var_value = var_value.reshape((var_value.shape[0] // 15, 15, var_value.shape[1]))
        date = np.genfromtxt(txt, dtype=np.int, skip_header=1)[:, 0]
        date = np.unique(date)
        NAME = np.append(NAME, var_name)

        # Don't exist date is generated, generated date value is nan #
        gen_data = []
        for gd in total_length:
            if gd in date:
                gen_data.append(var_value[date == gd][0])
            else:
                nan_array = np.full((15, len(var_name)), np.nan)
                gen_data.append(nan_array)
        VAR = np.concatenate((VAR, np.array(gen_data)), axis=2)

    del_name_idx = []
    for loc, nn in enumerate(NAME):
        for col in drop_col:
            if col in nn:
                del_name_idx = np.append(del_name_idx, loc)
    NAME = np.delete(NAME, del_name_idx)
    VAR = np.delete(VAR, del_name_idx, axis=2)
    VAR[VAR == -9999.] = np.nan
    VAR[VAR == -999.] = np.nan

    return NAME, total_length, VAR


def InputSet(dict, region, month, PM_name, train_date):
    global corr_criteria
    if PM_name == 'PM2_5': corr_criteria = 0.3
    if PM_name == 'PM10': corr_criteria = 0.2

    name, date, var = dict[region+'_name'], dict[region+'_date'], dict[region+'_data']
    name, date, var = np.array(name), np.array(date), np.array(var)

    # Extract month date #
    month_date = []
    for m in month:
        for md in train_date:
            if m == int(str(md)[4:6]): month_date.append(md)
            else: continue
    date_idx = []; none_date=[]
    for md in month_date:
        idx = np.where(date == md)[0]
        if idx.size != 0: date_idx.append(idx)
        else: none_date.append(md)
    date_idx = np.squeeze(np.array(date_idx, dtype=np.int))
    date, var = date[date_idx], var[date_idx]
    print("Train date shape : {} / Train data shape : {}".format(date.shape, var.shape))

    # This code is prepared to calculate the R and P-value #
    obs_idx, mod_idx = [], []
    for ii, nn in enumerate(name):
        if 'O_' in nn:
            obs_idx.append(ii)
        else:
            if ('1000hpa' in nn) or ('925hpa' in nn):
                continue
            else:
                mod_idx.append(ii)
    obs_name = name[obs_idx]
    mod_name = name[mod_idx]
    obs = var[:, 3:7, obs_idx]   # T04 ~ T06 : D+0, observation variable
    mod = var[:, 7:11, mod_idx]  # T07 ~ T10 : D+1, forecast variable
    pm_obs = var[:, 3:7, name == 'O_' + PM_name].reshape([-1])  # T04 ~ T06 : D+0, PM concentration
    pm_mod = var[:, 7:11, name == 'O_'+PM_name].reshape([-1])   # T07 ~ T10 : D+1, PM concentration

    # The R and p-value of observation variable is calculated #
    OBS_NAME = np.empty(0, np.str)
    for ii, nn in enumerate(obs_name):
        if nn in ['O_U', 'O_V', 'O_RN_ACC']:
            OBS_NAME = np.append(OBS_NAME, nn)
            continue
        vv = obs[:, :, ii].reshape([-1])
        isnan1 = np.where(np.isnan(vv) == True)[0]
        isnan2 = np.where(np.isnan(pm_obs) == True)[0]
        isnan = np.unique(np.concatenate((isnan1, isnan2)))
        pm = np.delete(pm_obs, isnan)
        vv = np.delete(vv, isnan)
        corr, pvalue = pearsonr(pm, vv)
        if np.abs(corr) >= corr_criteria and pvalue <= 0.01:
            OBS_NAME = np.append(OBS_NAME, nn)

    # The R and p-value of forecast variable is calculated #
    MOD_NAME = np.empty(0, np.str)
    for ii, nn in enumerate(mod_name):
        if 'cluster_pattern' in nn:
            MOD_NAME = np.append(MOD_NAME, nn)
            continue
        vv = mod[:, :, ii].reshape([-1])
        isnan1 = np.where(np.isnan(vv) == True)[0]
        isnan2 = np.where(np.isnan(pm_mod) == True)[0]
        isnan = np.unique(np.concatenate((isnan1, isnan2)))
        pm = np.delete(pm_mod, isnan)
        vv = np.delete(vv    , isnan)
        corr, pvalue = pearsonr(pm, vv)
        if np.abs(corr) >= corr_criteria and pvalue <= 0.01:
            MOD_NAME = np.append(MOD_NAME, nn)

    # Delete overlapped variable at pressure level #
    for met_var in ['U', 'V', 'W', 'Ta', 'RH', 'gpm']:
        f_hpa, cs_hpa = np.empty(0, np.str), np.empty(0, np.str)
        for nn in MOD_NAME:
            if ('f_' in nn) and ('hpa' in nn) and (met_var in nn): f_hpa = np.append(f_hpa , nn)
            if ('cs_' in nn) and ('hpa' in nn) and (met_var in nn): cs_hpa = np.append(cs_hpa, nn)
        hpa1, hpa2 = np.empty(0, np.int), np.empty(0, np.int)
        for fhpa in f_hpa:
            hpa1 = np.append(hpa1, int((fhpa.split('_')[1])[:-3]))
        for cshpa in cs_hpa:
            hpa2 = np.append(hpa2, int((cshpa.split('_')[1])[:-3]))
        # Except for the lowest altitude variable, the overlapping variables are deleted #
        if len(hpa1) > 1:
            delete_hpa_idx = np.argsort(hpa1)[::-1][1:]
            delete_f_hpa = f_hpa[delete_hpa_idx]
            for del_f_hpa in delete_f_hpa:
                MOD_NAME = np.delete(MOD_NAME, np.squeeze(np.where(MOD_NAME == del_f_hpa)))
        if len(hpa2) > 1:
            delete_hpa_idx = np.argsort(hpa2)[::-1][1:]
            delete_cs_hpa = cs_hpa[delete_hpa_idx]
            for del_cs_hpa in delete_cs_hpa:
                MOD_NAME = np.delete(MOD_NAME, np.squeeze(np.where(MOD_NAME == del_cs_hpa)))

    return np.concatenate((OBS_NAME, MOD_NAME))


def Read(pm, rt, region, main_region, data_dict, season, month, day_n, train_date, test_date, print_log):
    name = np.array(data_dict[region + '_name'])
    var = np.array(data_dict[region + '_data'])
    date = np.array(data_dict[region + '_date'])
    use_col = np.array(data_dict[main_region + '_{}_use_col'.format(season)])

    # Extraction of selected input names #
    name_idx = np.empty(0, np.int)
    for idx, nn in enumerate(name):
        if nn in use_col:
            name_idx = np.append(name_idx, idx)
    var = var[:, :, name_idx]
    name = name[name_idx]

    # Target and cmaq value #
    tar_idx = np.where(name == 'O_{}'.format(pm))[0]
    tar = var[:, MODEL_TIMES[rt]['T_OBS']:, tar_idx]
    cmq_idx = np.where(name == 'f_{}'.format(pm))[0]
    cmq = var[:, MODEL_TIMES[rt]['T_OBS']:, cmq_idx]

    # Separated observation and model value; var1 is observation, var2 is model #
    var1, var2, name1, name2 = PrePost(var=var, name=name, separation=MODEL_TIMES[rt]['T_OBS'])

    # Separate all period to training and test period #
    train1, test1, _, _ = Separate(var=var1, dat=date, train=train_date, test=test_date, month=month)
    train2, test2, _, _ = Separate(var=var2, dat=date, train=train_date, test=test_date, month=month)
    cmq_train, cmq_test, _, _ = Separate(var=cmq, dat=date, train=train_date, test=test_date, month=month)
    tar_train, tar_test, dat_train, dat_test = Separate(var=tar, dat=date, train=train_date, test=test_date, month=month)
    train_len, test_len = len(dat_train), len(dat_test)

    train2 = DaySeparate(day_n=day_n, dlen=MODEL_TIMES[rt]['D0_LEN'], var=train2)
    test2 = DaySeparate(day_n=day_n, dlen=MODEL_TIMES[rt]['D0_LEN'], var=test2)
    tar_train = DaySeparateTarget(day_n=day_n, dlen=MODEL_TIMES[rt]['D0_LEN'], var=tar_train)
    cmq_train = DaySeparateTarget(day_n=day_n, dlen=MODEL_TIMES[rt]['D0_LEN'], var=cmq_train)
    tar_test = DaySeparateTarget(day_n=day_n, dlen=MODEL_TIMES[rt]['D0_LEN'], var=tar_test)
    cmq_test = DaySeparateTarget(day_n=day_n, dlen=MODEL_TIMES[rt]['D0_LEN'], var=cmq_test)

    # Remove nan #
    train1, train2, tar_train, cmq_train, dat_train = RemoveNan(in1=train1, in2=train2, tar=tar_train, cmq=cmq_train, dat=dat_train)
    test1, test2, tar_test, cmq_test, dat_test = RemoveNan(in1=test1, in2=test2, tar=tar_test, cmq=cmq_test, dat=dat_test)
    # print("Except train & test len :: {}, {} / Total train & test len :: {}, {}".format(train_len-len(dat_train), test_len-len(dat_test), len(dat_train), len(dat_test)))

    if print_log:
        if not region == main_region:
            print("Append train data :: PM={} / RunTime={} / SubArea={} / ExceptTrain & AppendTrain={} & {}".format(pm, rt, region, train_len-len(dat_train), len(dat_train)))
        else:
            print("Main train data :: PM={} / RunTime={} / Area={} / ExceptTrain & Train={} & {}".format(pm, rt, main_region, train_len-len(dat_train), len(dat_train)))
    else: pass
    return train1, train2, tar_train, cmq_train, dat_train, test1, test2, tar_test, cmq_test, dat_test, name1, name2


def PrePost(var, name, separation):
    obs_idx = np.empty(0, dtype=np.int)
    prd_idx = np.empty(0, dtype=np.int)
    for idx, nn in enumerate(name):
        if 'O_' in nn:
            obs_idx = np.append(obs_idx, idx)
        else:
            prd_idx = np.append(prd_idx, idx)
    pre = var[:, :separation, obs_idx]
    post = var[:, separation:, prd_idx]
    pre_name = name[obs_idx]
    post_name = name[prd_idx]
    return pre, post, pre_name, post_name


def Separate(var, dat, train, test, month):
    train_month_date, test_month_date = [], []
    for m in month:
        for tmd in train:
            if m == int(str(tmd)[4:6]): train_month_date.append(tmd)
        for tmd in test:
            if m == int(str(tmd)[4:6]): test_month_date.append(tmd)

    train_month_idx, test_month_idx = np.empty(0, np.int), np.empty(0, np.int)
    for tmd in sorted(train_month_date):
        train_month_idx = np.append(train_month_idx, np.squeeze(np.where(dat == tmd)))
    for tmd in sorted(test_month_date):
        test_month_idx = np.append(test_month_idx, np.squeeze(np.where(dat == tmd)))

    train_date, train_var = dat[train_month_idx], var[train_month_idx]
    test_date, test_var = dat[test_month_idx], var[test_month_idx]

    return train_var, test_var, train_date, test_date


def DaySeparate(day_n, dlen, var):
    if day_n == 'D0': var = var[:, :dlen, :]
    if day_n == 'D1': var = var[:, :dlen+4, :]
    if day_n == 'D2': var = var[:, :, :]

    return var


def DaySeparateTarget(day_n, dlen, var):
    if day_n == 'D0': var = var[:, :dlen, :]
    if day_n == 'D1': var = var[:, dlen:dlen+4, :]
    if day_n == 'D2': var = var[:, dlen+4:, :]

    return var


def RemoveNan(in1, in2, tar, cmq, dat):
    nan1 = np.unique(np.where(np.isnan(in1) == True)[0])
    nan2 = np.unique(np.where(np.isnan(in2) == True)[0])
    nan3 = np.unique(np.where(np.isnan(tar) == True)[0])
    nan4 = np.unique(np.where(np.isnan(cmq) == True)[0])
    nan_box = np.array([], dtype=np.int)
    if len(nan1) != 0: nan_box = np.append(nan_box, nan1)
    if len(nan2) != 0: nan_box = np.append(nan_box, nan2)
    if len(nan3) != 0: nan_box = np.append(nan_box, nan3)
    if len(nan4) != 0: nan_box = np.append(nan_box, nan4)
    nan_box = np.unique(nan_box)
    if len(nan_box) != 0:
        in1 = np.delete(in1, nan_box, axis=0)
        in2 = np.delete(in2, nan_box, axis=0)
        tar = np.delete(tar, nan_box, axis=0)
        cmq = np.delete(cmq, nan_box, axis=0)
        dat = np.delete(dat, nan_box, axis=0)

    return in1, in2, tar, cmq, dat


def Normalization(train, val):
    min = np.nanmin(train, axis=1)
    max = np.nanmax(train, axis=1)
    train_nor = np.zeros(train.shape, dtype=np.float)
    val_nor = np.zeros(val.shape  , dtype=np.float)
    for ii in range(train.shape[-1]):
        for ss in range(train.shape[1]):
            train_nor[:, ss, ii] = (train[:, ss, ii] - min[:, ii]) / (max[:, ii] - min[:, ii])
        for ss in range(val.shape[1]):
            val_nor[:, ss, ii] = (val[:, ss, ii] - min[:, ii]) / (max[:, ii] - min[:, ii])

    return train_nor, val_nor, min, max


def MakeNormalizedFiles(path, name, min, max):
    name = np.concatenate((['MinMax', 'time'], name))
    with open(path, 'w') as f:
        for nn in name:
            f.write("".join(nn)+" ")
        f.write('\n')
        for ix, mn in enumerate(min):
            f.write("".join('min')+" ")
            f.write("".join(str(ix+1))+" ")
            for kk in mn:
                f.write("".join(str(kk))+" ")
            f.write('\n')
        for ix, mx in enumerate(max):
            f.write("".join('max')+" ")
            f.write("".join(str(ix+1))+" ")
            for kk in mx:
                f.write("".join(str(kk))+" ")
            f.write('\n')
    f.close()
    print('Make min max file : '+path)


def Noise(train1_std, train2_std, target_std, x1, x2, y, batch_size):
    np.random.seed(42)
    rnd = np.random.uniform(0.0, 1.0, batch_size)
    X1 = np.zeros(shape=x1.shape, dtype=x1.dtype)
    X2 = np.zeros(shape=x2.shape, dtype=x2.dtype)
    Y = np.zeros(shape=y.shape, dtype=y.dtype)
    for ii in range(batch_size):
        X1[:, ii, :] = x1[:, ii, :] + train1_std * rnd[ii]
        X2[:, ii, :] = x2[:, ii, :] + train2_std * rnd[ii]
        Y[:, ii, :] = y[:, ii, :] + target_std * rnd[ii]
    return X1, X2, Y


def Directory(path, remove, make):
    if remove:
        if type(path) is list:
            for p in path:
                if os.path.exists(path=p): shutil.rmtree(path=p)
                print("Remove directory : {}".format(p))
        else:
            if os.path.exists(path=path): shutil.rmtree(path=path)
            print("Remove directory : {}".format(path))

    if make:
        if type(path) is list:
            for p in path:
                if not os.path.isdir(p):
                    os.makedirs(name=p)
                    print("Make directory : {}".format(p))
        else:
            if not os.path.isdir(path):
                os.makedirs(name=path)
                print("Make directory : {}".format(path))


class EarlyStopping:
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss :
            self._step += 1
            if self._step >= self.patience:
                if self.verbose:
                    return True
        else:
            self._step = 0
            self._loss = loss
        return False


def FileSave(epoch, save_weight_path, test_weight_path):
    file1 = save_weight_path+'-'+str(epoch)+'.data-00000-of-00001'
    file2 = save_weight_path+'-'+str(epoch)+'.index'
    file3 = save_weight_path+'-'+str(epoch)+'.meta'

    if os.path.exists(test_weight_path):
        for path_, dirs, files in os.walk(test_weight_path):
            for file in files:
                os.remove(os.path.join(path_, file))

    new_file1 = test_weight_path+'save0.data-00000-of-00001'
    new_file2 = test_weight_path+'save0.index'
    new_file3 = test_weight_path+'save0.meta'

    shutil.move(file1, new_file1)
    shutil.move(file2, new_file2)
    shutil.move(file3, new_file3)

    if os.path.exists(save_weight_path):
        for path_, dirs, files in os.walk(save_weight_path):
            for file in files:
                os.remove(os.path.join(path_, file))


def FileDelete(path):
    if os.path.exists(path):
        for path_, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(path_, file))


def InverseNormalization(path):
    is_minmax = np.genfromtxt(path, dtype=np.str, usecols=0, skip_header=1)
    name = np.genfromtxt(path, dtype=np.str)[0]
    minmax = np.genfromtxt(path, dtype=np.float, skip_header=1)
    name, minmax = name[2:], minmax[:, 2:]
    min_idx = np.where(is_minmax == 'min')[0]
    max_idx = np.where(is_minmax == 'max')[0]
    min, max = minmax[min_idx, :], minmax[max_idx, :]
    return min, max, name


def PreRead(pm, rt, region, data_dict, season, month, day_n, predict_date):
    # Set data name, value, date #
    name = data_dict[region + '_name']
    var = data_dict[region + '_data']
    date = data_dict[region + '_date']
    use_col = np.array(data_dict[region + '_{}_use_col'.format(season)])
    name, var, date = np.array(name), np.array(var), np.array(date)
    # var[var == -9999.] = np.nan

    # Extraction of selected input names #
    name_idx = np.array([idx for idx, nn in enumerate(name) if nn in use_col], dtype=np.int)
    var = var[:, :, name_idx]
    name = name[name_idx]

    # Target and cmaq value #
    tar_idx = np.where(name == 'O_{}'.format(pm))[0]
    tar = var[:, MODEL_TIMES[rt]['T_OBS']:, tar_idx]
    cmq_idx = np.where(name == 'f_{}'.format(pm))[0]
    cmq = var[:, MODEL_TIMES[rt]['T_OBS']:, cmq_idx]

    # Separated observation and model value; var1 is observation, var2 is model #
    var1, var2, name1, name2 = PrePost(var=var, name=name, separation=MODEL_TIMES[rt]['T_OBS'])

    # Separate all period to test period #
    pre_month_date = []
    for m in month:
        for pmd in predict_date:
            if m == int(str(pmd)[4:6]): pre_month_date.append(pmd)
    pre_month_idx = np.empty(0, np.int)
    for tmd in sorted(pre_month_date):
        pre_month_idx = np.append(pre_month_idx, np.squeeze(np.where(date == tmd)))
    pre_date, pre_var1, pre_var2 = date[pre_month_idx], var1[pre_month_idx], var2[pre_month_idx]
    pre_tar, pre_cmq = tar[pre_month_idx], cmq[pre_month_idx]

    # Split day separate #
    pre_var2 = DaySeparate(day_n=day_n, dlen=MODEL_TIMES[rt]['D0_LEN'], var=pre_var2)
    pre_tar = DaySeparateTarget(day_n=day_n, dlen=MODEL_TIMES[rt]['D0_LEN'], var=pre_tar)
    pre_cmq = DaySeparateTarget(day_n=day_n, dlen=MODEL_TIMES[rt]['D0_LEN'], var=pre_cmq)

    # Remove nan #
    '''
    nan_idx = np.empty(0, np.int)
    for vv in [pre_var1, pre_var2, pre_tar, pre_cmq]:
        nan = np.unique(np.where(np.isnan(vv) == True)[0])
        nan_idx = np.append(nan_idx, nan)
    nan_idx = np.sort(np.unique(nan_idx))

    pre_var1, pre_var2 = np.delete(pre_var1, nan_idx, 0), np.delete(pre_var2, nan_idx, 0)
    pre_tar, pre_cmq = np.delete(pre_tar, nan_idx, 0), np.delete(pre_cmq, nan_idx, 0)
    pre_dat = np.delete(pre_date, nan_idx, 0)
    '''

    return pre_var1, pre_var2, pre_tar, pre_cmq, pre_date


def CategoryEvaluation(df, model):
    # Make category data #
    for col in ['Target', model]:
        df[col] = df[col].mask(df[col].astype('float') < 15.5, other=1)
        df[col] = df[col].mask((df[col].astype('float') >= 15.5) & (df[col].astype('float') < 35.5), other=2)
        df[col] = df[col].mask((df[col].astype('float') >= 35.5) & (df[col].astype('float') < 75.5), other=3)
        df[col] = df[col].mask(df[col].astype('float') >= 75.5, other=4)

    # Count of each category #
    GG = df[(df['Target'] == 1) & (df[model] == 1)]
    MM = df[(df['Target'] == 2) & (df[model] == 2)]
    BB = df[(df['Target'] == 3) & (df[model] == 3)]
    VV = df[(df['Target'] == 4) & (df[model] == 4)]
    POD_down = df[(df['Target'] >= 3) & (df[model] >= 1)]
    POD_up   = df[(df['Target'] >= 3) & (df[model] >= 3)]
    FAR_down = df[(df['Target'] >= 1) & (df[model] >= 3)]
    FAR_up   = df[(df['Target'] <= 2) & (df[model] >= 3)]

    # Ratio #
    ACC_up, ACC_down = len(GG) + len(MM) + len(BB) + len(VV), len(df)
    POD_up, POD_down = len(POD_up), len(POD_down)
    FAR_up, FAR_down = len(FAR_up), len(FAR_down)

    # Category evaluation #
    ACC = (ACC_up / ACC_down) * 100
    try:
        POD = (POD_up / POD_down) * 100
    except ZeroDivisionError:
        POD = 0
    try:
        FAR = (FAR_up / FAR_down) * 100
    except ZeroDivisionError:
        FAR = 0
    try:
        F1  = 2 * ((POD * (100-FAR)) / (POD + (100-FAR)))
    except ZeroDivisionError:
        F1 = 0

    return ACC, ACC_up, ACC_down, POD, POD_up, POD_down, FAR, FAR_up, FAR_down, F1


def CategoryEvaluationExcess50(df, model):
    # Make category data #
    for col in ['Target', model]:
        df[col] = df[col].mask(df[col].astype('float') < 15.5, other=1)
        df[col] = df[col].mask((df[col].astype('float') >= 15.5) & (df[col].astype('float') < 50.5), other=2)
        df[col] = df[col].mask((df[col].astype('float') >= 50.5) & (df[col].astype('float') < 75.5), other=3)
        df[col] = df[col].mask(df[col].astype('float') >= 75.5, other=4)

    # Count of each category #
    GG = df[(df['Target'] == 1) & (df[model] == 1)]
    MM = df[(df['Target'] == 2) & (df[model] == 2)]
    BB = df[(df['Target'] == 3) & (df[model] == 3)]
    VV = df[(df['Target'] == 4) & (df[model] == 4)]
    POD_down = df[(df['Target'] >= 3) & (df[model] >= 1)]
    POD_up   = df[(df['Target'] >= 3) & (df[model] >= 3)]
    FAR_down = df[(df['Target'] >= 1) & (df[model] >= 3)]
    FAR_up   = df[(df['Target'] <= 2) & (df[model] >= 3)]

    # Ratio #
    ACC_up, ACC_down = len(GG) + len(MM) + len(BB) + len(VV), len(df)
    POD_up, POD_down = len(POD_up), len(POD_down)
    FAR_up, FAR_down = len(FAR_up), len(FAR_down)

    # Category evaluation #
    ACC = (ACC_up / ACC_down) * 100
    try:
        POD = (POD_up / POD_down) * 100
    except ZeroDivisionError:
        POD = 0
    try:
        FAR = (FAR_up / FAR_down) * 100
    except ZeroDivisionError:
        FAR = 0
    try:
        F1  = 2 * ((POD * (100-FAR)) / (POD + (100-FAR)))
    except ZeroDivisionError:
        F1 = 0

    return ACC, ACC_up, ACC_down, POD, POD_up, POD_down, FAR, FAR_up, FAR_down, F1


def StatisticalEvaluation(df, model):
    Target = df['Target'].astype('float')
    Model  = df[model].astype('float')
    AvgTarget, AvgModel = (sum(Target) / len(df)), (sum(Model) / len(df))

    # R value #
    gradient, intercept, R_value, p_value, std_err = stats.linregress(Target, Model)

    # IOA value #
    IOA_up, IOA_down = sum((Model - Target)**2), sum((abs(Model - AvgTarget) + abs(Target - AvgTarget))**2)
    IOA_val = 1 - (IOA_up / IOA_down)

    # RMSE value #
    MSE = mean_squared_error(Target, Model)
    RMSE_val = math.sqrt(MSE)

    # NMB value #
    NMB_up, NMB_down = sum(Model - Target), sum(Target)
    NMB_val = (NMB_up / NMB_down) * 100
    
    return R_value, IOA_val, RMSE_val, NMB_val
