import pandas as pd
import numpy as np

base_path = 'G:/dataset/a3d6_chusai_a_train/dealt_data/'
trainset1_path = base_path + 'dataset_train_1/'
testset1_path = base_path + 'dataset_test_1/'
trainset2_path = base_path + 'dataset_train_2/'
testset2_path = base_path + 'dataset_test_2/'
trainset3_path = base_path + 'dataset_train_3/'

train_file = base_path + 'train_test/train.csv'
validate_file = base_path + 'train_test/validate.csv'
test_file = base_path + 'train_test/test.csv'

register = 'register.csv'
launch = 'launch.csv'
create = 'create.csv'
activity = 'activity.csv'


def get_train_label(train_path, test_path):
    train_reg = pd.read_csv(train_path + register, usecols=['user_id'])
    train_lau = pd.read_csv(train_path + launch, usecols=['user_id'])
    train_cre = pd.read_csv(train_path + create,usecols=['user_id'])
    train_act = pd.read_csv(train_path + activity, usecols=['user_id'])
    train_data_id = np.unique(pd.concat([train_reg, train_lau,train_cre, train_act]))

    test_reg = pd.read_csv(test_path + register, usecols=['user_id'])
    test_lau = pd.read_csv(test_path + launch, usecols=['user_id'])
    test_cre = pd.read_csv(test_path + create, usecols=['user_id'])
    test_act = pd.read_csv(test_path + activity, usecols=['user_id'])
    test_data_id = np.unique(pd.concat([test_reg, test_lau, test_cre, test_act]))

    train_label = []
    for id in train_data_id:
        if id in test_data_id:
            train_label.append(1)
        else:
            train_label.append(0)
    train_data = pd.DataFrame()
    train_data['user_id'] = train_data_id
    train_data['label'] = train_label
    return train_data


def get_test(test_path):
    test_reg = pd.read_csv(test_path + register, usecols=['user_id'])
    test_lau = pd.read_csv(test_path + launch, usecols=['user_id'])
    test_cre = pd.read_csv(test_path + create, usecols=['user_id'])
    test_act = pd.read_csv(test_path + activity, usecols=['user_id'])
    test_data_id = np.unique(pd.concat([test_reg, test_lau, test_cre, test_act]))

    test_data = pd.DataFrame()
    test_data['user_id'] = test_data_id
    return test_data


def get_register_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['register_type'] = list(row['register_type'])[0]
    feature['device_type'] = list(row['device_type'])[0]
    return feature


def get_launch_feature(lau):
    grouped = lau['launch_day'].groupby(lau['user_id'])
    lau_feature = pd.DataFrame()
    lau_feature['user_id'] = lau['user_id'].unique()
    lau_feature['lau_times'] = list(grouped.count())
    lau_feature['lau_day_max'] = list(grouped.max())
    lau_feature['lau_day_min'] = list(grouped.min())
    lau_feature['lau_day_diff'] = list(grouped.max() - grouped.min())
    lau_feature['lau_day_mean'] = list(grouped.mean())
    lau_feature['lau_day_std'] = list(grouped.std())
    # print(lau_feature.head())
    # lau_feature.fillna(0)
    return lau_feature


def get_create_feature(cre):
    grouped = cre['create_day'].groupby(cre['user_id'])
    cre_feature = pd.DataFrame()
    cre_feature['user_id'] = cre['user_id'].unique()
    cre_feature['cre_times'] = list(grouped.count())
    cre_feature['cre_day_max'] = list(grouped.max())
    cre_feature['cre_day_min'] = list(grouped.min())
    cre_feature['cre_day_diff'] = list(grouped.max() - grouped.min())
    cre_feature['cre_day_mean'] = list(grouped.mean())
    cre_feature['cre_day_std'] = list(grouped.std())
    # print(lau_feature.head())
    # cre_feature.fillna(0)

    cre1 = cre.copy()
    cre1['cre_times'] = np.ones(len(cre1))
    # print(cre1.head())
    grouped = cre1.groupby(['user_id', 'create_day'])
    cre_feature1 = pd.DataFrame()
    cre_feature1['cre_day_numbers'] = list(grouped.size())
    # print(grouped.count())

    cre2 = cre1.copy()
    cre2.drop_duplicates(subset=['user_id', 'create_day'], keep='first', inplace=True)
    cre2 = cre2.reset_index(drop=True)
    cre_feature1['user_id'] = cre2['user_id']
    cre_feature1 = cre_feature1[['user_id', 'cre_day_numbers']]
    # 创建视频的天数特征
    cre_feature2 = pd.DataFrame()
    grouped = cre_feature1['cre_day_numbers'].groupby(cre_feature1['user_id'])
    cre_feature2['user_id'] = cre_feature1['user_id'].unique()
    cre_feature2['cre_daynum'] = list(grouped.count())
    cre_feature2['cre_daynum_max'] = list(grouped.max())
    cre_feature2['cre_daynum_min'] = list(grouped.min())
    cre_feature2['cre_daynum_diff'] = list(grouped.max() - grouped.min())
    cre_feature2['cre_daynum_mean'] = list(grouped.mean())
    cre_feature2['cre_daynum_std'] = list(grouped.std())
    # print(cre_feature2.head())
    feature = pd.merge(cre_feature, cre_feature2, on='user_id', how='left')
    return feature


def get_activity_feature(act):
    grouped = act['action_day'].groupby(act['user_id'])
    act_feature = pd.DataFrame()
    act_feature['user_id'] = act['user_id'].unique()
    act_feature['act_times'] = list(grouped.count())
    act_feature['act_day_max'] = list(grouped.max())
    act_feature['act_day_min'] = list(grouped.min())
    act_feature['act_day_diff'] = list(grouped.max() - grouped.min())
    act_feature['act_day_mean'] = list(grouped.mean())
    act_feature['act_day_std'] = list(grouped.std())
    # print(act_feature.head())
    print('activity 表特征1 提取完毕')

    act1 = act.copy()
    act1['act_times'] = np.ones(len(act1))
    # print(cre1.head())
    grouped = act1.groupby(['user_id', 'action_day'])
    act_feature1 = pd.DataFrame()
    act_feature1['act_day_numbers'] = list(grouped.size())
    # print(grouped.count())

    act2 = act1.copy()
    # del act2['page', 'video_id', 'author_id', 'action_type']
    # print(act2.head())
    act2.drop_duplicates(subset=['user_id', 'action_day'], keep='first', inplace=True)
    act2 = act2.reset_index(drop=True)
    # print(act2[:10])
    act_feature1['user_id'] = act2['user_id']
    act_feature1 = act_feature1[['user_id', 'act_day_numbers']]
    # print(act_feature1[:10])

    act_feature2 = pd.DataFrame()

    grouped = act_feature1['act_day_numbers'].groupby(act_feature1['user_id'])
    act_feature2['user_id'] = act_feature1['user_id'].unique()
    act_feature2['act_daynum'] = list(grouped.count())
    act_feature2['act_daynum_max'] = list(grouped.max())
    act_feature2['act_daynum_min'] = list(grouped.min())
    act_feature2['act_daynum_diff'] = list(grouped.max() - grouped.min())
    act_feature2['act_daynum_mean'] = list(grouped.mean())
    act_feature2['act_daynum_std'] = list(grouped.std())
    feature = pd.merge(act_feature, act_feature2, on='user_id', how='left')
    # print(act_feature2[:10])
    print('activity 表特征2提取完毕')

    act1 = act.copy()
    act1 = act1[(act1['action_type'] < 4)]
    act1['act_times'] = np.ones(len(act1))
    # print(cre1.head())
    grouped = act1['video_id'].groupby(act1['user_id'])
    tmp_feature = pd.DataFrame()
    tmp_feature['act_video_num'] = list(grouped.size())

    act3 = act1.copy()
    # del act2['page', 'video_id', 'author_id', 'action_type']
    # print(act2.head())
    act3.drop_duplicates(subset=['user_id', 'video_id'], keep='first', inplace=True)
    act3 = act3.reset_index(drop=True)
    # print(act3[:10])
    tmp_feature['user_id'] = act3['user_id']
    tmp_feature = tmp_feature[['user_id', 'act_video_num']]
    # print(tmp_feature[:10])

    act_feature3 = pd.DataFrame()
    grouped = tmp_feature['act_video_num'].groupby(tmp_feature['user_id'])
    act_feature3['user_id'] = tmp_feature['user_id'].unique()
    act_feature3['act_videonum'] = list(grouped.count())
    act_feature3['act_videonum_max'] = list(grouped.max())
    act_feature3['act_videonum_min'] = list(grouped.min())
    # act_feature3['act_videonum_diff'] = list(grouped.max() - grouped.min())
    act_feature3['act_videonum_mean'] = list(grouped.mean())
    act_feature3['act_videonum_std'] = list(grouped.std())
    # print(act_feature3[:10])
    print('activity 表特征3提取完毕')
    feature = pd.merge(feature, act_feature3, on='user_id', how='left')

    act1 = act.copy()
    act1 = act1[(act1['action_type'] < 4)]
    act1['act_times'] = np.ones(len(act1))
    # print(cre1.head())
    grouped = act1['action_day'].groupby(act1['author_id'], sort=True)
    tmp_feature = pd.DataFrame()
    tmp_feature['act_actioned_daynum'] = list(grouped.size())

    act3 = act1.copy()
    # del act2['page', 'video_id', 'author_id', 'action_type']
    # print(act2.head())
    act3.drop_duplicates(subset=['author_id', 'action_day'], keep='first', inplace=True)
    act3 = act3.reset_index(drop=True)
    # print(act3[:10])
    tmp_feature['user_id'] = act3['author_id']
    tmp_feature = tmp_feature[['user_id', 'act_actioned_daynum']]
    # print(tmp_feature[:10])

    act_feature4 = pd.DataFrame()

    grouped = tmp_feature['act_actioned_daynum'].groupby(tmp_feature['user_id'])
    act_feature4['user_id'] = tmp_feature['user_id'].unique()
    act_feature4['actioned_daynum'] = list(grouped.count())
    act_feature4['actioned_daynum_max'] = list(grouped.max())
    act_feature4['actioned_daynum_min'] = list(grouped.min())
    # act_feature3['act_videonum_diff'] = list(grouped.max() - grouped.min())
    act_feature4['actioned_daynum_mean'] = list(grouped.mean())
    act_feature4['actioned_daynum_std'] = list(grouped.std())
    # print(act_feature4[:10])
    print('activity 表特征4 提取完毕')
    feature = pd.merge(feature, act_feature4, on='user_id', how='left')

    act1 = act.copy()
    act1 = act1[(act1['action_type'] < 4)]
    act1['act_times'] = np.ones(len(act1))
    # print(cre1.head())
    grouped = act1['user_id'].groupby(act1['author_id'], sort=True)
    tmp_feature = pd.DataFrame()
    tmp_feature['act_actioned_usernum'] = list(grouped.size())

    act3 = act1.copy()
    # del act2['page', 'video_id', 'author_id', 'action_type']
    # print(act2.head())
    act3.drop_duplicates(subset=['author_id', 'user_id'], keep='first', inplace=True)
    act3 = act3.reset_index(drop=True)
    # print(act3[:10])
    tmp_feature['user_id'] = act3['author_id']
    tmp_feature = tmp_feature[['user_id', 'act_actioned_usernum']]
    # print(tmp_feature[:10])
    act_feature5 = pd.DataFrame()

    grouped = tmp_feature['act_actioned_usernum'].groupby(tmp_feature['user_id'], sort=True)
    act_feature5['user_id'] = tmp_feature['user_id'].unique()
    act_feature5['actioned_usernum'] = list(grouped.count())
    act_feature5['actioned_usernum_max'] = list(grouped.max())
    act_feature5['actioned_usernum_min'] = list(grouped.min())
    # act_feature3['act_videonum_diff'] = list(grouped.max() - grouped.min())
    act_feature5['actioned_usernum_mean'] = list(grouped.mean())
    act_feature5['actioned_usernum_std'] = list(grouped.std())
    # print(act_feature5[:10])
    print('activity 表特征5 提取完毕')
    feature = pd.merge(feature, act_feature5, on='user_id', how='left')

    grouped = act['action_day'].groupby(act['author_id'], sort=True)
    act_feature6 = pd.DataFrame()
    act_feature6['user_id'] = act['author_id'].unique()
    act_feature6['actioned_times'] = list(grouped.count())
    act_feature6['actioned_day_max'] = list(grouped.max())
    act_feature6['actioned_day_min'] = list(grouped.min())
    act_feature6['actioned_day_diff'] = list(grouped.max() - grouped.min())
    act_feature6['actioned_day_mean'] = list(grouped.mean())
    act_feature6['actioned_day_std'] = list(grouped.std())
    # print(act_feature6[:10])
    print('activity 表特征6 提取完毕')
    return feature


def get_eachday_feature(df, dfname, datespace, begin, user_index, day_index):
    user_matrix = dict()
    for i in df.index:
        u = df.loc[i].values[user_index]
        d = df.loc[i].values[day_index]
        if u not in user_matrix:
            user_matrix[u] = [0 for x in range(datespace)]
            user_matrix[u][d - begin] = 1

    arr = []
    for k, v in user_matrix.items():
        row = [k]
        row.extend(v)
        arr.append(row)
    cols = ['user_id']
    for i in range(datespace):
        cols.append(dfname + str(i + 1))
    ret = pd.DataFrame(list(arr), columns=cols)
    # print(ret.head())
    # print(len(ret))
    return ret


def feature_extract(path, user_id, begin):
    reg = pd.read_csv(path + register)
    lau = pd.read_csv(path + launch)
    cre = pd.read_csv(path + create)
    act = pd.read_csv(path + activity)

    feature = pd.DataFrame()
    feature['user_id'] = user_id
    # 注册类型，设备特征
    reg_feature = reg.groupby('user_id', sort=True).apply(get_register_feature)
    feature = pd.merge(feature, pd.DataFrame(reg_feature), on='user_id', how='left')
    print('register 表特征提取完毕')
    # print(feature.head())

    # launch feature
    lau_feature = get_launch_feature(lau)
    feature = pd.merge(feature, lau_feature, on='user_id', how='left')
    # print(feature.head())
    print('launch 表特征提取完毕')

    # 创建视频的次数特征
    cre_feature = get_create_feature(cre)
    feature = pd.merge(feature, cre_feature, on='user_id', how='left')
    # print(feature.head())
    print('create 表特征提取完毕')

    # action特征
    act_feature = get_activity_feature(act)
    feature = pd.merge(feature, act_feature, on='user_id', how='left')
    print('activity 表特征提取完毕')

    datespace = 16
    user_index = 0
    day_index = 1
    lau_day_feature = get_eachday_feature(lau, 'lau', datespace, begin, user_index, day_index)
    feature = pd.merge(feature, lau_day_feature, on='user_id', how='left')
    cre_day_feature = get_eachday_feature(cre, 'cre', datespace, begin, user_index, day_index)
    feature = pd.merge(feature, cre_day_feature, on='user_id', how='left')
    act_day_feature = get_eachday_feature(act, 'act', datespace, begin, user_index, day_index)
    feature = pd.merge(feature, act_day_feature, on='user_id', how='left')
    acted_day_feature = get_eachday_feature(act, 'acted', datespace, begin, 4, day_index)
    feature = pd.merge(feature, acted_day_feature, on='user_id', how='left')

    feature.fillna(0, inplace=True)
    print(feature[:10])
    return feature


def get_data_feature():
    train1 = get_train_label(trainset1_path, testset1_path)
    feature1 = feature_extract(trainset1_path, train1['user_id'], 1)
    feature1['label'] = train1['label'].fillna(0)
    print('第一组训练数据特征提取完毕')

    train2 = get_train_label(trainset2_path, testset2_path)
    feature2 = feature_extract(trainset2_path, train2['user_id'], 8)
    feature2['label'] = train2['label'].fillna(0)
    print('第二组训练数据特征提取完毕')

    # train_feature = pd.concat([feature1, feature2])
    # train_feature.to_csv(train_file, index=False)
    feature1.to_csv(train_file, index=False)
    feature2.to_csv(validate_file, index=False)
    print('训练数据存储完毕')

    test_data = get_test(trainset3_path)
    test_feature = feature_extract(trainset3_path, test_data['user_id'], 15)
    test_feature.to_csv(test_file, index=False)
    print('测试数据存储完毕')


get_data_feature()