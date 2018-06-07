import numpy as np
users = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_register_log.txt', dtype=str, delimiter='\t')

launch_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/app_launch_log.txt',
                        dtype=str, delimiter='\t')
video_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/video_create_log.txt',
                        dtype=str, delimiter='\t')
activity_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_activity_log.txt',
                        dtype=str, delimiter='\t')

user_activity = [0 for x in range(30)]
a_day_u = dict()
for i in range(30):
    day_u[i] = []
for u in activity_log:
    i = u[0]
    d = int(u[1]) - 1
    if i not in day_u[d]:
        day_u[d].append(i)