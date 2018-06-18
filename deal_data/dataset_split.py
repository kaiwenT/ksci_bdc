import pandas as pd

base_path = 'G:/dataset/a3d6_chusai_a_train/dealt_data/'
trainset1_path = base_path + 'dataset_train_1/'
testset1_path = base_path + 'dataset_test_1/'
trainset2_path = base_path + 'dataset_train_2/'
testset2_path = base_path + 'dataset_test_2/'
trainset3_path = base_path + 'dataset_train_3/'

launch = pd.read_csv(base_path + 'app_launch_log.csv')
register = pd.read_csv(base_path + 'user_register_log.csv')
create = pd.read_csv(base_path + 'video_create_log.csv')
activity = pd.read_csv(base_path + 'user_activity_log.csv')


def cut_data_between(dataset_path, begin_day, end_day):
    temp_register = register[(register['register_day'] >= begin_day) & (register['register_day'] <= end_day)]
    temp_launch = launch[(launch['launch_day'] >= begin_day) & (launch['launch_day'] <= end_day)]
    temp_create = create[(create['create_day'] >= begin_day) & (create['create_day'] <= end_day)]
    temp_activity = activity[(activity['action_day'] >= begin_day) & (activity['action_day'] <= end_day)]

    temp_register.to_csv(dataset_path + 'register.csv', index=False)
    temp_launch.to_csv(dataset_path + 'launch.csv', index=False)
    temp_create.to_csv(dataset_path + 'create.csv', index=False)
    temp_activity.to_csv(dataset_path + 'activity.csv', index=False)


def generate_dataset():
    print('开始划分数据集...')

    begin_day = 1
    end_day = 16
    cut_data_between(trainset1_path, begin_day, end_day)
    begin_day = 17
    end_day = 23
    cut_data_between(testset1_path, begin_day, end_day)
    print('训练集1，测试集1划分完成')

    begin_day = 8
    end_day = 23
    cut_data_between(trainset2_path, begin_day, end_day)
    begin_day = 24
    end_day = 30
    cut_data_between(testset2_path, begin_day, end_day)
    print('训练集2，测试集2划分完成')

    begin_day = 15
    end_day = 30
    cut_data_between(trainset3_path, begin_day, end_day)
    print('训练集3 划分完成')


generate_dataset()
