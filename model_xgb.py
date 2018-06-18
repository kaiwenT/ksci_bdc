import xgboost as xgb
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def train(x, y):
    dtrain = xgb.DMatrix(data=x, label=y)

    # learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    # param_grid = dict(learning_rate=learning_rate)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    # grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    # specify parameters via map
    param = {'max_depth': 10, 'eta': 0.3, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 2
    bst = xgb.train(param, dtrain, num_round)
    bst.save_model('G:/models/ksci_bdc_model/model_xgboost/002.model')

    xgb.plot_importance(bst)
    plt.show()


def evaluate(data, label):
    dtrain = xgb.DMatrix(data=data)
    # make prediction
    bst = xgb.Booster(model_file='G:/models/ksci_bdc_model/model_xgboost/002.model')
    pred_y = bst.predict(dtrain)
    preds = []
    for x in pred_y:
        if x >= 0.4:
            preds.append(1)
        else:
            preds.append(0)
    precision = metrics.precision_score(label, preds, average='binary')
    recall = metrics.recall_score(label, preds, average='binary')
    f1score = metrics.f1_score(label, preds, average='binary')

    print('precision: ', precision)
    print('recall: ', recall)
    print('F1-Score: ', f1score)

    return f1score


def predict(test_data, fname):
    dtrain = xgb.DMatrix(data=test_data)
    # make prediction
    bst = xgb.Booster(model_file='G:/models/ksci_bdc_model/model_xgboost/002.model')
    pred_y = bst.predict(dtrain)
    pre = []
    for x in pred_y:
        if x >= 0.4:
            pre.append(1)
        else:
            pre.append(0)

    user_ids = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/user_ids', dtype=str)
    active_users = []
    for i in range(len(pre)):
        if pre[i] == 1:
            active_users.append(user_ids[i])

    train_user_ids = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/train_user_ids', dtype=str)
    labels = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/user_label', dtype=int)
    for i in range(len(labels)):
        if labels[i] == 1 and train_user_ids[i] not in active_users:
            active_users.append(train_user_ids[i])

    np.savetxt('G:/dataset/a3d6_chusai_a_train/prediction/prediction_xgb' + fname + '.txt', active_users, '%s')


# read in data
train_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/train_feature', dtype=float)
train_y = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/user_label', dtype=int)
#
train(train_x, train_y)
score = evaluate(train_x, train_y)


test_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/test_feature', dtype=float)
predict(test_x, str(score))
