import pandas as pd

base_path = 'G:/dataset/a3d6_chusai_a_train/'
new_path = 'G:/dataset/a3d6_chusai_a_train/dealt_data/'

app_launch_log = pd.read_csv(base_path + 'app_launch_log.txt', sep='\t', header=None)
app_launch_log = app_launch_log.sort_values(by=[0, 1])
app_launch_log = app_launch_log.rename({0: 'user_id', 1: 'launch_day'}, axis=1)
app_launch_log.to_csv(new_path + 'app_launch_log.csv', index=False)

user_register_log = pd.read_csv(base_path + 'user_register_log.txt', sep='\t', header=None)
user_register_log = user_register_log.sort_values(by=[0, 1])
user_register_log = user_register_log.rename({0: 'user_id', 1: 'register_day',
                                           2: 'register_type', 3: 'device_type'}, axis=1)
user_register_log.to_csv(new_path + 'user_register_log.csv', index=False)

video_create_log = pd.read_csv(base_path + 'video_create_log.txt', sep='\t', header=None)
video_create_log = video_create_log.sort_values(by=[0, 1])
video_create_log = video_create_log.rename({0: 'user_id', 1: 'create_day'}, axis=1)
video_create_log.to_csv(new_path + 'video_create_log.csv', index=False)

user_activity_log = pd.read_csv(base_path + 'user_activity_log.txt', sep='\t', header=None)
user_activity_log = user_activity_log.sort_values(by=[0, 1])
user_activity_log = user_activity_log.rename({0: 'user_id', 1: 'action_day',
                                           2: 'page', 3: 'video_id',
                                           4: 'author_id', 5: 'action_type'}, axis=1)
user_activity_log.to_csv(new_path + 'user_activity_log.csv', index=False)

