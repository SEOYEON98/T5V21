#
# ==============================================================================
#   Program Name   : Evaluation
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
import numpy as np
import pandas as pd
from Module.Config import MODEL_TIMES, PMS, ABS_PATH
from Module.Utils import ConcatData, InverseNormalization, PreRead, TotalDate, Directory, CategoryEvaluation, StatisticalEvaluation

#
#
#
EVAL_YEAR = 2024
TimeEval    = True
DayEval     = True
SectionEval = True
CONC_PATH = os.path.join(ABS_PATH, 'Result', 'TestResult', '{{PM}}') + os.sep
RESULT_PATH = os.path.join(ABS_PATH, 'Result', 'Evaluation', '{{PM}}') + os.sep
for PM in PMS[0:1]:
    if PM == 'PM2_5': pm_name = 'PM2.5'
    if PM == 'PM10' : pm_name = 'PM10'
    ConcPath = CONC_PATH.replace('{{PM}}', pm_name)
    ResultPath = RESULT_PATH.replace('{{PM}}', pm_name)
    Directory(path=ResultPath, remove=False, make=True)

    for RT in sorted(MODEL_TIMES.keys())[1:2]:

        for A in ['SU']:
            print("Evaluation :: PM={} / RunTime={} / Area={}".format(PM, RT, A))
            # Load data #
            CONC_df = pd.read_csv(ConcPath + "{}_{}_{}.txt".format(A, RT, EVAL_YEAR), delimiter="\t")

            # NaN value check #
            nan_date_df = pd.DataFrame([])
            for col in ['Target', 'AI', 'CMAQ']:
                nan_df = CONC_df[CONC_df[col].isnull()]
                nan_date_df = pd.concat([nan_date_df, nan_df])
            nan_date = nan_date_df['Date'].unique()

            # Drop NaN date #
            for nd in nan_date:
                CONC_df = CONC_df.drop(CONC_df[CONC_df['Date'] == nd].index, inplace=False).reset_index(drop=True)

            # Evaluation of T-step results #
            if TimeEval:
                T_FILE = open(ResultPath + "{}_{}_TIME_EVAL_{}.txt".format(A, RT, EVAL_YEAR), 'w')
                T_FILE.write("Model" + '\t' + "Time" + '\t' +  "RMSE" + '\t' +  "R" + '\t' +  "IOA" + "\t" + "NMB" + "\t"
                             "ACC" + '\t' + "ACC_ratio" + '\t' + "POD" + '\t' + "POD_ratio" + '\t' + "FAR" + '\t' + "FAR_ratio" + '\t' + "F1_score" + "\n")
                for T in [i for i in range(5, 6)]:
                    T_CONC_df = CONC_df.loc[(CONC_df['Time'] == T)].reset_index(drop=True)

                    for m in ['AI', 'CMAQ']:
                        R, IOA, RMSE, NMB = StatisticalEvaluation(df=T_CONC_df, model=m)
                        ACC, ACC_up, ACC_down, POD, POD_up, POD_down, FAR, FAR_up, FAR_down, F1 = CategoryEvaluation(df=T_CONC_df, model=m)
                        T_FILE.write("{:s}".format(m) + '\t' + "{:d}".format(T) + '\t' +
                                     "{:.1f}".format(RMSE) + '\t' + "{:.2f}".format(R) + '\t' + "{:.2f}".format(IOA) + '\t' + "{:.2f}".format(NMB) + "\t" +
                                     "{:.1f}".format(ACC) + '\t' + "{:d}//{:d}".format(ACC_up, ACC_down) + '\t' +
                                     "{:.1f}".format(POD) + '\t' + "{:d}//{:d}".format(POD_up, POD_down) + '\t' +
                                     "{:.1f}".format(FAR) + '\t' + "{:d}//{:d}".format(FAR_up, FAR_down) + '\t' +
                                     "{:.1f}".format(F1)  + "\n")
                T_FILE.close()

            #
            # Day average #
            #
            if DayEval:
                D_FILE = open(ResultPath + "{}_{}_DAY_EVAL_{}.txt".format(A, RT, EVAL_YEAR), 'w')
                D_FILE.write("Model" + '\t' + "Time" + '\t' +  "RMSE" + '\t' +  "R" + '\t' +  "IOA" + '\t' + "NMB" + "\t" +
                             "ACC" + '\t' + "ACC_ratio" + '\t' + "POD" + '\t' + "POD_ratio" + '\t' + "FAR" + '\t' + "FAR_ratio" + '\t' + "F1_score" + "\n")
                for DAY in ['D0', 'D1', 'D2']:
                    if DAY == 'D0': T_LIST = [i for i in range(5 , 8)]
                    if DAY == 'D1': T_LIST = [i for i in range(8 , 12)]
                    if DAY == 'D2': T_LIST = [i for i in range(12, 16)]

                    # Extract data contained in T_LIST #
                    DAY_df = CONC_df.loc[(CONC_df['Time'] >= T_LIST[0]) & (CONC_df['Time'] <= T_LIST[-1])]

                    # Data average #
                    day_target, day_ai, day_cmaq = [], [], []
                    date = DAY_df['Date'].unique()
                    for d in date:
                        avg_df = DAY_df.loc[(DAY_df['Date'] == d)].drop(labels=['Time'], axis=1).reset_index(drop=True)
                        day_target.append('{:.1f}'.format(avg_df['Target'].mean()))
                        day_ai.append('{:.1f}'.format(avg_df['AI'].mean()))
                        day_cmaq.append('{:.1f}'.format(avg_df['CMAQ'].mean()))
                    DAY_CONC_df = pd.DataFrame({'Date': date, 'Target': day_target, 'AI': day_ai, 'CMAQ': day_cmaq})
                    DAY_CONC_df = DAY_CONC_df[['Date', 'Target', 'AI', 'CMAQ']]
                    DAY_CONC_df.to_csv(ConcPath + "{}_{}_{}_{}.txt".format(A, DAY, RT, EVAL_YEAR), sep="\t", index=False)

                    # Evaluation of day results #
                    for m in ['AI', 'CMAQ']:
                        R, IOA, RMSE, NMB = StatisticalEvaluation(df=DAY_CONC_df, model=m)
                        ACC, ACC_up, ACC_down, POD, POD_up, POD_down, FAR, FAR_up, FAR_down, F1 = CategoryEvaluation(df=DAY_CONC_df.copy(), model=m)
                        D_FILE.write("{}".format(m) + '\t' + "{}".format(DAY) + '\t' + 
                                     "{:.1f}".format(RMSE) + '\t' + "{:.2f}".format(R) + '\t' + "{:.2f}".format(IOA) + '\t' + "{:.2f}".format(NMB) + "\t" +
                                     "{:.1f}".format(ACC) + '\t' + "{:d}//{:d}".format(ACC_up, ACC_down) + '\t' + 
                                     "{:.1f}".format(POD) + '\t' + "{:d}//{:d}".format(POD_up, POD_down) + '\t' + 
                                     "{:.1f}".format(FAR) + '\t' + "{:d}//{:d}".format(FAR_up, FAR_down) + '\t' + 
                                     "{:.1f}".format(F1)  + "\n")
                D_FILE.close()

            #
            # Section evaluation #
            #
            if SectionEval:
                D_FILE = open(ResultPath + "{}_{}_Section_DAY_EVAL_{}.txt".format(A, RT, EVAL_YEAR), 'w')
                D_FILE.write("Model" + '\t' + "Time" + '\t' +  "Section" + "\t" + "RMSE" + '\t' +  "R" + '\t' +  "IOA" + '\t' + "NMB" + "\t" +
                             "ACC" + '\t' + "ACC_ratio" + "\n")

                for DAY in ['D0', 'D1', 'D2']:
                    if DAY == 'D0': T_LIST = [i for i in range(5 , 8)]
                    if DAY == 'D1': T_LIST = [i for i in range(8 , 12)]
                    if DAY == 'D2': T_LIST = [i for i in range(12, 16)]

                    # Extract data contained in T_LIST #
                    DAY_df = CONC_df.loc[(CONC_df['Time'] >= T_LIST[0]) & (CONC_df['Time'] <= T_LIST[-1])]

                    # Data average #
                    day_target, day_ai, day_cmaq = [], [], []
                    date = DAY_df['Date'].unique()
                    for d in date:
                        avg_df = DAY_df.loc[(DAY_df['Date'] == d)].drop(labels=['Time'], axis=1).reset_index(drop=True)
                        day_target.append('{:.1f}'.format(avg_df['Target'].mean()))
                        day_ai.append('{:.1f}'.format(avg_df['AI'].mean()))
                        day_cmaq.append('{:.1f}'.format(avg_df['CMAQ'].mean()))
                    DAY_CONC_df = pd.DataFrame({'Date': date, 'Target': day_target, 'AI': day_ai, 'CMAQ': day_cmaq})
                    DAY_CONC_df = DAY_CONC_df[['Date', 'Target', 'AI', 'CMAQ']]

                    # Evaluation of each section #
                    for section in ['Good', 'Moderate', 'Bad+VeryBad']:
                        if section == 'Good':
                            SEC_CONC = DAY_CONC_df.loc[DAY_CONC_df['Target'].astype('float') < 15.5].reset_index(drop=True)
                        elif section == 'Moderate':
                            SEC_CONC = DAY_CONC_df.loc[(DAY_CONC_df['Target'].astype('float') >= 15.5) & (DAY_CONC_df['Target'].astype('float') < 35.5)].reset_index(drop=True)
                        else:
                            SEC_CONC = DAY_CONC_df.loc[(DAY_CONC_df['Target'].astype('float') >= 35.5)].reset_index(drop=True)

                        # Evaluation of day results #
                        for m in ['AI', 'CMAQ']:
                            R, IOA, RMSE, NMB = StatisticalEvaluation(df=SEC_CONC, model=m)
                            ACC, ACC_up, ACC_down, _, _, _, _, _, _, _ = CategoryEvaluation(df=SEC_CONC.copy(), model=m)
                            D_FILE.write("{}".format(m) + '\t' + "{}".format(DAY) + '\t' + "{}".format(section) + "\t"
                                         "{:.1f}".format(RMSE) + '\t' + "{:.2f}".format(R) + '\t' + "{:.2f}".format(IOA) + '\t' + "{:.2f}".format(NMB) + "\t" +
                                         "{:.1f}".format(ACC) + '\t' + "{:d}//{:d}".format(ACC_up, ACC_down) + "\n")
                D_FILE.close()


