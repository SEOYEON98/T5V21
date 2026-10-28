#
# ==============================================================================
#   Program Name   : TrainCode
# ------------------------------------------------------------------------------
#   Description
# ------------------------------------------------------------------------------
#   Company Name   : NIER (National Institute of Environmental Research)
#   Create Date    : 2022-04-13
# ------------------------------------------------------------------------------
#   Update History : 2022-06-10
# ==============================================================================
#

# -------------------------
# Set library.
#
import os
import sys
import pickle
import datetime
import numpy as np
from Module.Config import AREA_GROUP, season_month, DeviceNumber, ABS_PATH, SAVE_PATH, MINMAX_PATH, LOG_PATH, batch_size, state_sizes, MODEL_TIMES, counts
# Device selection is optional and must precede the TensorFlow import.
if DeviceNumber is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = DeviceNumber
import tensorflow as tf
from Module.Utils import TotalDate, Read, Normalization, MakeNormalizedFiles, Directory, EarlyStopping, Noise, FileSave, FileDelete
from Module.Model import LSTM, RMSE

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Receive arguments #
PM = sys.argv[1]
RT = sys.argv[2]
A = sys.argv[3]
season = sys.argv[4]
DAY_N = sys.argv[5]
month = season_month[season]

# PM name for directory name #
if PM == 'PM2_5': PM_dn = 'PM2.5'
if PM == 'PM10': PM_dn = 'PM10'

# Define group area #
if A in AREA_GROUP['SMA']: areas = AREA_GROUP['SMA']
if A in AREA_GROUP['CC']: areas = AREA_GROUP['CC']
if A in AREA_GROUP['YN']: areas = AREA_GROUP['YN']
if A in AREA_GROUP['HN']: areas = AREA_GROUP['HN']
if A in AREA_GROUP['GWE']: areas = AREA_GROUP['GWE']
if A in AREA_GROUP['JJ']: areas = AREA_GROUP['JJ']

# Set train & test date #
_, train_date = TotalDate(start_date='20160101', end_date='20201231')
_, test_date = TotalDate(start_date='20210101', end_date='20211231')

# Set log #
LogPath = LOG_PATH.replace('{{RunTime}}', RT).replace('{{PM}}', PM_dn)
Log = open(LogPath + '{}_TrainLog.txt'.format(A), 'a')

# Data load #
DICT_PATH = os.path.join(ABS_PATH, 'Data', 'ConcatData') + os.sep
with open(DICT_PATH + "data_{}h.pickle".format(RT), "rb") as fr:
    data = pickle.load(fr)

# Make Train & validation data #
for LOC, GA in enumerate(areas):
    if LOC == 0:
        trn1, trn2, tar, cmq, tdate, _, _, _, _, _, nm1, nm2 = Read(pm=PM, rt=RT, region=GA, main_region=A, data_dict=data, season=season, month=month, day_n=DAY_N, train_date=train_date, test_date=test_date, print_log=True)
    else:
        sub_trn1, sub_trn2, sub_tar, sub_cmq, sub_tdate, _, _, _, _, _, sub_nm1, sub_nm2 = Read(pm=PM, rt=RT, region=GA, main_region=A, data_dict=data, season=season, month=month, day_n=DAY_N, train_date=train_date, test_date=test_date, print_log=True)
        trn1 = np.append(trn1, sub_trn1, axis=0)
        trn2 = np.append(trn2, sub_trn2, axis=0)
        tar = np.append(tar, sub_tar, axis=0)
        cmq = np.append(cmq, sub_cmq, axis=0)
_, _, _, _, _, tt1, tt2, tt_tar, tt_cmq, tt_date, _, _ = Read(pm=PM, rt=RT, region=A, main_region=A, data_dict=data, season=season, month=month, day_n=DAY_N, train_date=train_date, test_date=test_date, print_log=False)
print('Train data shape ::: obs={} / model={} / target={} / cmaq={}'.format(trn1.shape, trn2.shape, tar.shape, cmq.shape))
print('Test data shape ::: obs={} / model={} / target={} / cmaq={}'.format(tt1.shape, tt2.shape, tt_tar.shape, tt_cmq.shape))
print("Train parameter ::: PM={} / ModelTime={} / Area={} / Season={} / TargetDay={} / Train data shape=obs:{}&model:{}".format(PM, RT, A, season, DAY_N, trn1.shape, trn2.shape), file=Log)
# Data transpose :: [BatchSize, SequenceLength, Variable] --> [SequenceLength, BatchSize, Variabels] #
trn1 = trn1.transpose([1, 0, 2])
trn2 = trn2.transpose([1, 0, 2])
tar = tar.transpose([1, 0, 2])
cmq = cmq.transpose([1, 0, 2])
tt1 = tt1.transpose([1, 0, 2])
tt2 = tt2.transpose([1, 0, 2])
tt_tar = tt_tar.transpose([1, 0, 2])
tt_cmq = tt_cmq.transpose([1, 0, 2])
print(trn1.shape)


# Normalization #
norm_trn1, norm_tt1, min1, max1 = Normalization(trn1, tt1)
norm_trn2, norm_tt2, min2, max2 = Normalization(trn2, tt2)
norm_tar, norm_tt_tar, mint, maxt = Normalization(tar, tt_tar)

# Make normalized files (encoder, decoder, and target) #
MinMaxPath = MINMAX_PATH.replace('{{PM}}', PM_dn).replace('{{RunTime}}', RT).replace('{{Area}}', A).replace('{{Season}}', season).replace('{{Day}}', DAY_N)
SavePath = SAVE_PATH.replace('{{PM}}', PM_dn).replace('{{RunTime}}', RT).replace('{{Area}}', A).replace('{{Season}}', season).replace('{{Day}}', DAY_N)
Directory(path=[MinMaxPath, SavePath], remove=False, make=True)
MakeNormalizedFiles(MinMaxPath + 'Encoder.txt', nm1, min1, max1)
MakeNormalizedFiles(MinMaxPath + 'Decoder.txt', nm2, min2, max2)
MakeNormalizedFiles(MinMaxPath + 'Targets.txt', [PM], mint, maxt)

# standard deviation for noise Calculation #
std1, std2, std3 = np.std(norm_trn1, axis=1), np.std(norm_trn2, axis=1), np.std(norm_tar, axis=1)

# Data length #
train_size, test_size = norm_trn1.shape[1], norm_tt1.shape[1]
seq_length1, seq_length2, seq_length3 = norm_trn1.shape[0], norm_trn2.shape[0], norm_tar.shape[0]
num_input1, num_input2 = norm_trn1.shape[-1], norm_trn2.shape[-1]

# Data of test set is copied #
Te1 = np.array([norm_tt1] * batch_size, dtype=np.float32).transpose(2, 1, 0, 3)
Te2 = np.array([norm_tt2] * batch_size, dtype=np.float32).transpose(2, 1, 0, 3)

# Make save directory #
save_weight_path = SavePath + 'SaveWeightFiles/'
test_weight_path = SavePath + 'TestWeightFiles/'
Directory(path=save_weight_path, remove=True, make=True)
Directory(path=test_weight_path, remove=False, make=True)

# RNN model graph #
tf.reset_default_graph()
input1 = tf.placeholder(tf.float32, shape=[seq_length1, batch_size, num_input1], name='EncoderInput')
input2 = tf.placeholder(tf.float32, shape=[seq_length2, batch_size, num_input2], name='DecoderInput')
target = tf.placeholder(tf.float32, shape=[seq_length3, batch_size, 1] , name='Target')
lr = tf.placeholder(tf.float32, shape=[], name='LearningRate')

lstm = LSTM(input1, input2, state_sizes, DAY_N, MODEL_TIMES[RT]['D0_LEN'], seq_length3)
logits = lstm.predict()
loss = RMSE(logits, target)
optimize = tf.train.AdamOptimizer(lr)
gradients, variables = zip(*optimize.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 1)
optimizer = optimize.apply_gradients(zip(gradients, variables))

# Training :: Learning Rate Scheduling #
count_loss = []
for count in range(counts):
    TEST_LOSS, offsets = [], []
    early_stop_length = 200*(count+1)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=early_stop_length+1)
    early_stopping = EarlyStopping(patience=early_stop_length, verbose=1)
    epoch, max_step, rate = 0, (train_size//batch_size) * 100000, 0.0001 * 10 ** (-1 * count)
    print('Count :'+str(count)+', Lr :'+str(rate))
    print('Count :'+str(count)+', Lr :'+str(rate), file=Log)

    # Restore in LRS Process #
    if count > 0:
        saver.restore(sess, test_weight_path+'save0')

    for step in range(max_step):
        offset = (step * batch_size) % (train_size - batch_size)
        offsets.append(offset)

        # Data split from batch size #
        batch_1 = norm_trn1[:, offset:(offset + batch_size), :]
        batch_2 = norm_trn2[:, offset:(offset + batch_size), :]
        batch_tar = norm_tar[:, offset:(offset + batch_size),:]
        batch_1, batch_2, batch_tar = Noise(train1_std=std1, train2_std=std2, target_std=std3, x1=batch_1, x2=batch_2, y=batch_tar, batch_size=batch_size)

        # Input #
        feed_dict = {input1: batch_1, input2: batch_2, target: batch_tar, lr: rate}
        _, l = sess.run([optimizer, loss], feed_dict=feed_dict)

        if (step != 0) and (offsets[step-1] > offsets[step]):
            permutation_ = np.random.permutation(train_size)
            norm_trn1 = norm_trn1[:, permutation_, :]
            norm_trn2 = norm_trn2[:, permutation_, :]
            norm_tar = norm_tar[:, permutation_, :]

            result = np.zeros_like(norm_tt_tar)
            for ii in range(test_size):
                batch_1 = Te1[ii]
                batch_2 = Te2[ii]
                feed_dict_te = {input1: batch_1, input2: batch_2}
                lg = sess.run([logits], feed_dict=feed_dict_te)
                lg = np.mean(np.array(lg)[0, :, :, :], axis=1)
                result[:, ii, :] = lg
            lo = np.sqrt(np.mean((result-norm_tt_tar)**2))
            TEST_LOSS.append(lo)

            if epoch % 50 == 0:
                print('Epoch: %i, Test loss: %4f, Minimum: %.4f' %(epoch, lo, np.min(TEST_LOSS)))
                print('Epoch: %i, Test loss: %4f, Minimum: %.4f' %(epoch, lo, np.min(TEST_LOSS)), file=Log)

            saver.save(sess, save_weight_path, global_step=epoch)
            epoch += 1

            if early_stopping.validate(lo):
               print('Training process is stopped early....')
               print('Training process is stopped early....', file=Log)
               break

    saved_epoch = epoch - early_stop_length -1
    count_loss.append(TEST_LOSS[saved_epoch])

    if count == 0:
        FileSave(epoch=saved_epoch, save_weight_path=save_weight_path, test_weight_path=test_weight_path)
        print('%i count file save, Test loss: %4f' %(count, TEST_LOSS[saved_epoch]))
        print('%i count file save, Test loss: %4f' %(count, TEST_LOSS[saved_epoch]), file=Log)

    if count != 0:
        if count_loss[count] < count_loss[count-1]:
            FileSave(epoch=saved_epoch, save_weight_path=save_weight_path, test_weight_path=test_weight_path)
            print('%i count file save, Test loss: %4f' %(count, TEST_LOSS[saved_epoch]))
            print('%i count file save, Test loss: %4f' %(count, TEST_LOSS[saved_epoch]), file=Log)

        else:
            FileDelete(path=save_weight_path)
            print('%i count dont save, Process termination ...' % count)
            print('%i count dont save, Process termination ...' % count, file=Log)
            break

