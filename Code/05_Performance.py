#
# ==============================================================================
#   Program Name   : Performance
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
import shutil
import pickle
import numpy as np
from Module.Config import MODEL_TIMES, PM_DIR, ABS_PATH, SAVE_PATH, MINMAX_PATH, season_box, month_box, batch_size, state_sizes, DeviceNumber
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Optional device selection must precede the TensorFlow import.
if DeviceNumber is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = DeviceNumber
import tensorflow as tf
from Module.Utils import ConcatData, InverseNormalization, PreRead, TotalDate, Directory
from Module.Model import LSTM

""" SET PARAMETERS """
PM = 'PM2.5'
AREAS = ['SU']
target_year = 2024


# ---------------------------------
# Set path.
#
result_file_path = os.path.join(ABS_PATH, 'Result', 'TestResult', '{{PM}}') + os.sep

# ---------------------------------
# Set test date.
#
_, predict_date = TotalDate(start_date='20240101', end_date='20240331')

# Predict #
for RT in sorted(MODEL_TIMES.keys())[1:2]:
    # Data load #
    DICT_PATH = os.path.join(ABS_PATH, 'Data', 'ConcatData') + os.sep
    with open(DICT_PATH + "data_{}h.pickle".format(RT), "rb") as fr:
        data_dict = pickle.load(fr)

    for A in AREAS:
        # Empty list for result #
        DATE, TIME, OBS, RNN, CMQ = [], [], [], [], []

        # Result file name #
        Directory(path=result_file_path.replace('{{PM}}', PM), remove=False, make=True)
        result_file_name = '{}_{}_{}.txt'.format(A, RT, target_year)
        result_file = result_file_path.replace('{{PM}}', PM) + result_file_name

        # If already exist result file, this is deleted #
        if os.path.isfile(result_file): os.remove(result_file)
        f = open(result_file, 'w')
        f.write('Date' + '\t' + 'Time' + '\t' + 'Target' + '\t' + 'AI' + '\t' + 'CMAQ' + '\n')

        for season, month in zip(season_box, month_box):

            for day_n in ['D0', 'D1', 'D2']:
                SavePath = SAVE_PATH.replace('{{PM}}', PM).replace('{{RunTime}}', RT).replace('{{Area}}', A).replace('{{Season}}', season).replace('{{Day}}', day_n)
                MinMaxPath = MINMAX_PATH.replace('{{PM}}', PM).replace('{{RunTime}}', RT).replace('{{Area}}', A).replace('{{Season}}', season).replace('{{Day}}', day_n)
                if day_n == 'D0': day = np.arange(MODEL_TIMES[RT]['D0_LEN']) + MODEL_TIMES[RT]['T_OBS'] + 1
                if day_n == 'D1': day = [8, 9, 10, 11]
                if day_n == 'D2': day = [12, 13, 14, 15]

                # Make prediction data #
                var1, var2, tar, cmq, dat = PreRead(pm=PM_DIR[PM]['PM_NAME'], rt=RT, region=A, data_dict=data_dict, season=season, month=month, day_n=day_n, predict_date=predict_date)

                if dat.shape[0] != 0:
                    print("Predict :: RunTime={} / Area={} / Season={} / Month={} / Day={}".format(RT, A, season, month, day_n))
                    # Load min max data #
                    min1, max1, en_name = InverseNormalization(path=MinMaxPath+'Encoder.txt')
                    min2, max2, de_name = InverseNormalization(path=MinMaxPath + 'Decoder.txt')
                    mint, maxt, _ = InverseNormalization(path=MinMaxPath + 'Targets.txt')

                    # Normalization #
                    var1 = (var1-min1)/(max1-min1)
                    var2 = (var2-min2)/(max2-min2)

                    # Data transpose :: [BatchSize, SequenceLength, Variable] --> [SequenceLength, BatchSize, Variable] #
                    var1, var2, tar, cmq = var1.transpose(1, 0, 2), var2.transpose(1, 0, 2), tar.transpose(1, 0, 2), cmq.transpose(1, 0, 2)

                    # Data length #
                    seq_length1 = var1.shape[0]
                    seq_length2 = var2.shape[0]
                    seq_length3 = tar.shape[0]
                    num_input1 = var1.shape[-1]
                    num_input2 = var2.shape[-1]

                    #
                    # LSTM model graph #
                    #
                    tf.reset_default_graph()
                    input1 = tf.placeholder(tf.float32, shape=[seq_length1, batch_size, num_input1], name='EncoderInput')
                    input2 = tf.placeholder(tf.float32, shape=[seq_length2, batch_size, num_input2], name='DecoderInput')
                    lstm = LSTM(input1, input2, state_sizes, day_n, MODEL_TIMES[RT]['D0_LEN'], seq_length3)
                    logits = lstm.predict()

                    # Save results of predict Set #
                    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                    sess.run(tf.global_variables_initializer())
                    saver = tf.train.Saver()
                    saver.restore(sess, SavePath + 'TestWeightFiles/save0')

                    result = np.zeros(shape=tar.shape, dtype=np.float)
                    for ii in range(tar.shape[1]):
                        batch_x1, batch_x2 = var1[:, ii, :], var2[:, ii, :]
                        batch_x1 = np.transpose([batch_x1 for _ in range(batch_size)], (1, 0, 2))
                        batch_x2 = np.transpose([batch_x2 for _ in range(batch_size)], (1, 0, 2))
                        feed_dict = {input1: batch_x1, input2: batch_x2}
                        lg = sess.run([logits], feed_dict=feed_dict)
                        lg = np.mean(np.squeeze(np.array(lg)), axis=-1).reshape([-1,1])
                        lg = lg * (maxt-mint) + mint
                        result[:, ii] = lg

                    for date_idx, dt in enumerate(dat):
                        for day_idx, dy in enumerate(day):
                            DATE.append(dt)
                            TIME.append(dy)
                            OBS.append(tar[day_idx, date_idx])
                            RNN.append(result[day_idx, date_idx])
                            CMQ.append(cmq[day_idx, date_idx])
                else:
                    print("Not included :: RunTime={} / Area={} / Season={} / Month={} / Day={}".format(RT, A, season, month, day_n))

        DATE, TIME, OBS, RNN, CMQ = np.array(DATE), np.array(TIME), np.array(OBS), np.array(RNN), np.array(CMQ)
        for ud in np.unique(DATE):
            for ut in np.unique(TIME):
                idx = np.where((DATE == ud) & (TIME == ut))[0]
                f.write('{:d}'.format(ud) + '\t' + '{:d}'.format(ut) + '\t' +
                        '{:.1f}'.format(np.nanmean(OBS[idx])) + '\t' + '{:.1f}'.format(np.nanmean(RNN[idx])) + '\t' + '{:.1f}'.format(np.nanmean(CMQ[idx])) + '\n')
        f.close()
        print("\n")
