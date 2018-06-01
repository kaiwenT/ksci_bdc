import numpy as np


dnn = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/prediction/prediction_dnn.txt', dtype=str)
cnn = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/prediction/prediction05_2.txt', dtype=str)
xgb = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/prediction/prediction_xgboost.txt', dtype=str)

total_ids = set()
common_ids = []

for x in dnn:
    total_ids.add(x)
for x in cnn:
    total_ids.add(x)
for x in xgb:
    total_ids.add(x)

for x in total_ids:
    if x in cnn and x in dnn and x in xgb:
        common_ids.append(x)

print(len(total_ids), len(common_ids))

np.savetxt('G:/dataset/a3d6_chusai_a_train/prediction/cnn_dnn_xgb_common' + '.txt', common_ids, '%s')
np.savetxt('G:/dataset/a3d6_chusai_a_train/prediction/cnn_dnn_xgb_sum' + '.txt', list(total_ids), '%s')