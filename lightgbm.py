import lightgbm as lgb
import numpy as np

train_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/train_feature', dtype=float)
train_y = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/user_label', dtype=int)

test_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/test_feature', dtype=float)

# lgb_train = lgb.Dataset(train_x, train_y)
# lgb_eval = lgb.Dataset(train_x, train_y, reference=lgb_train)

# 开始训练
print('设置参数')
params = {
    'boosting_type': 'gbdt',
    'boosting': 'dart',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': 3,
    'max_bin': 10,
    'min_data_in_leaf': 8,
    'feature_fraction': 0.6,
    'bagging_fraction': 1,
    'bagging_freq': 0,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'min_split_gain': 0
}

print("开始训练")
gbm = lgb.train(params,  # 参数字典
                [train_x, train_y],  # 训练集
                num_boost_round=2000,  # 迭代次数
                valid_sets=[train_x, train_y],  # 验证集
                early_stopping_rounds=30)  # 早停系数

### 线下预测
print("线下预测")
offline_test_X = lgb.Dataset(test_x, free_raw_data=False)
preds = gbm.predict(offline_test_X, num_iteration=gbm.best_iteration)  # 输出概率

user_ids = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/user_ids', dtype=str)
active_users = []
for i in range(len(preds)):
    if preds[i] >= 0.4:
        active_users.append(user_ids[i])

np.savetxt('G:/dataset/a3d6_chusai_a_train/prediction/prediction_feature1_lgb.txt', active_users, '%s')
