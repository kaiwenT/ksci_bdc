import xgboost as xgb
import numpy as np
import pandas as pd
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
    param = {'max_depth': 20, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 2
    bst = xgb.train(param, dtrain, num_round)
    bst.save_model('G:/models/ksci_bdc_model/model_xgboost/feature48.model')

    # xgb.plot_importance(bst)
    # plt.show()


def evaluate(data, label):
    dtrain = xgb.DMatrix(data=data)
    # make prediction
    bst = xgb.Booster(model_file='G:/models/ksci_bdc_model/model_xgboost/feature48.model')
    pred_y = bst.predict(dtrain)
    preds = []
    for x in pred_y:
        if x > 0.9:
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
    test_x = test_data.drop(['user_id'], axis=1)
    pred = pd.DataFrame()
    pred['user_id'] = test_data['user_id']
    # print(pred.head())
    dtrain = xgb.DMatrix(data=test_x)

    # do prediction
    bst = xgb.Booster(model_file='G:/models/ksci_bdc_model/model_xgboost/feature48.model')
    pred['prob'] = bst.predict(dtrain)
    # print(pred.head())
    T = 0.9
    ret = pred[(pred['prob'] > T)]['user_id']
    ret.to_csv(fname, index=False)


# read in data
train_xy = pd.read_csv('G:/dataset/a3d6_chusai_a_train/dealt_data/train_test/train.csv')
train_x = train_xy.drop(['user_id', 'label'], axis=1)
# train_xy['label'] = train_xy['label'].apply(lambda x: int(x))
# train_y = train_xy['label'].apply(lambda x: int(x))
train_xy['label'].fillna(1, inplace=True)
train_y = train_xy['label']
# print(train_x.head())
# print(train_y.head())

# train model
# train(train_x, train_y)
score = evaluate(train_x, train_y)

# predict
test_x = pd.read_csv('G:/dataset/a3d6_chusai_a_train/dealt_data/train_test/test.csv')
predict(test_x, 'G:/dataset/a3d6_chusai_a_train/prediction/predict_xgb.csv')
