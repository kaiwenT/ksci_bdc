import xgboost as xgb
import numpy as np
import sklearn.metrics as metrics


def train(x, y):
    dtrain = xgb.DMatrix(data=x, label=y)

    # specify parameters via map
    param = {'max_depth': 30, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 2
    bst = xgb.train(param, dtrain, num_round)
    bst.save_model('G:/models/ksci_bdc_model/model_xgboost/001.model')


def evaluate(data, label):
    dtrain = xgb.DMatrix(data=data)
    # make prediction
    bst = xgb.Booster(model_file='G:/models/ksci_bdc_model/model_xgboost/001.model')
    pred_y = bst.predict(dtrain)
    preds = [round(x) for x in pred_y]
    precision = metrics.precision_score(label, preds, average='binary')
    recall = metrics.recall_score(label, preds, average='binary')
    f1score = metrics.f1_score(label, preds, average='binary')

    print('precision: ', precision)
    print('recall: ', recall)
    print('F1-Score: ', f1score)


def predict(test_data):
    dtrain = xgb.DMatrix(data=test_data)
    # make prediction
    bst = xgb.Booster(model_file='G:/models/ksci_bdc_model/model_xgboost/001.model')
    pred_y = bst.predict(dtrain)
    pre = [round(x) for x in pred_y]

    user_ids = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_ids', dtype=str)
    active_users = []
    for i in range(len(pre)):
        if pre[i] == 1:
            active_users.append(user_ids[i])

    train_user_ids = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/train_user_ids', dtype=str)
    labels = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_label', dtype=int)
    for i in range(len(labels)):
        if labels[i] == 1 and train_user_ids[i] not in active_users:
            active_users.append(train_user_ids[i])

    np.savetxt('G:/dataset/a3d6_chusai_a_train/prediction/prediction_' + 'xgboost' + '.txt', active_users, '%s')


# read in data
# train_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_embedding', dtype=float)
# train_y = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_label', dtype=int)
#
# train(train_x, train_y)
# evaluate(train_x, train_y)

test_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/test_user_embedding', dtype=float)
predict(test_x)
