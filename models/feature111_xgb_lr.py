import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.externals import joblib


def train(X_train, y_train, X_test, y_test, T, c):
    # To avoid overfitting
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.3)
    model = xgb.XGBClassifier(nthread=4, learning_rate=0.1,
                              n_estimators=50, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)

    xgb_lr = LogisticRegression(C=0.5, penalty='l1')
    model.fit(X_train, y_train)
    joblib.dump(model, 'G:/models/ksci_bdc_model/model_xgboost/model_xgb.m')

    xgb_lr.fit(X_train_lr, y_train_lr)
    joblib.dump(xgb_lr, 'G:/models/ksci_bdc_model/model_xgboost/model_xgblr.m')
    y_xgb_lr_tmp = xgb_lr.predict_proba(X_test)[:, 1]
    # y_xgb_lr_test = np.round(y_xgb_lr_tmp)

    y_xgb_lr_test = []
    for x in y_xgb_lr_tmp:
        if x > T:
            y_xgb_lr_test.append(1)
        else:
            y_xgb_lr_test.append(0)
    # fpr_xgb_lr, tpr_xgb_lr, _ = roc_curve(y_test, y_xgb_lr_test)
    f1 = f1_score(y_test, y_xgb_lr_test)
    print("Xgboost + LR:", f1)
    # return fpr_xgb_lr, tpr_xgb_lr
    return f1


def predict(test, fname, t):
    test_data = test.drop(['user_id'], axis=1)
    test_data = np.array(test_data, dtype=np.float32)

    model = joblib.load('G:/models/ksci_bdc_model/model_xgboost/model_xgb.m')

    xgb_lr = joblib.load('G:/models/ksci_bdc_model/model_xgboost/model_xgblr.m')
    y_xgb_lr_tmp = xgb_lr.predict_proba(test_data)[:, 1]
    pred = pd.DataFrame()
    pred['user_id'] = test['user_id']
    pred['prob'] = y_xgb_lr_tmp
    ret = pred[(pred['prob'] > t)]['user_id']
    ret.to_csv(fname, index=False)


if __name__ == '__main__':
    # read in data
    train_xy = pd.read_csv('G:/dataset/a3d6_chusai_a_train/dealt_data/train_test/train.csv')
    train_x = train_xy.drop(['user_id', 'label'], axis=1)
    train_xy['label'].fillna(0, inplace=True)
    train_y = train_xy['label']

    validate_xy = pd.read_csv('G:/dataset/a3d6_chusai_a_train/dealt_data/train_test/validate.csv')
    validate_x = train_xy.drop(['user_id', 'label'], axis=1)
    validate_xy['label'].fillna(0, inplace=True)
    validate_y = train_xy['label']

    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # # plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
    # # plt.plot(fpr_grd_lr, tpr_grd_lr, label='GBT + LR')
    # plt.plot(fpr_xgboost, tpr_xgboost, label='XGB')
    # # plt.plot(fpr_lr, tpr_lr, label='LR')
    # # plt.plot(fpr_xgb_lr, tpr_xgb_lr, label='XGB + LR')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()

    # t = 0.3
    # max_t = t
    # max_f, max_c = 0, 0
    # while t < 0.61:
    #     c = 0.3
    #     while c < 0.81:
    #         score = train(train_x, train_y, validate_x, validate_y, t, c)
    #         c += 0.01
    #         if score > max_f:
    #             max_t = t
    #             max_c = c
    #             max_f = score
    #
    #     t += 0.01
    #
    # print('best threshold: %.2f  C: %.2f  F1-score: %.4f', max_t, max_c, max_f)
    # score = train(train_x, train_y, validate_x, validate_y, 0.48, 0.5)
    # # predict
    test_x = pd.read_csv('G:/dataset/a3d6_chusai_a_train/dealt_data/train_test/test.csv')
    #
    predict(test_x, 'G:/dataset/a3d6_chusai_a_train/prediction/predict_xgblr' + str(0.801) + '.txt', 0.48)
