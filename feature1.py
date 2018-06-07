import numpy as np
import data_process2 as dp2


def get_user_between(start, end):
    train_user = []

    users = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_register_log.txt', dtype=str, delimiter='\t')

    for u in users:
        day = int(u[1])
        if day in range(start, end + 1):
            train_user.append(u[0])

    launch_log = dp2.get_launch_log_between(start, end)
    video_log = dp2.get_video_log_between(start, end)
    activity_log = dp2.get_activity_log_between(start, end)

    for a in launch_log:
        u = a[0]
        if u not in train_user:
            train_user.append(u)

    for a in video_log:
        u = a[0]
        if u not in train_user:
            train_user.append(u)

    for a in activity_log:
        u = a[0]
        if u not in train_user:
            train_user.append(u)
    np.savetxt('G:/dataset/a3d6_chusai_a_train/feature_1/user' + str(start) + '-' + str(end), train_user, '%s')


def generate_traindata(start, end):
    train_users = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/user' + str(start) + '-' + str(end), dtype=str)

    user_label = dict()
    user_embedding = dict()
    for u in train_users:
        user_label[u] = 0
        user_embedding[u] = [0] * (7 * 3)
    launch_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/app_launch_log.txt',
                            dtype=str, delimiter='\t')
    video_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/video_create_log.txt',
                           dtype=str, delimiter='\t')
    activity_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_activity_log.txt',
                              dtype=str, delimiter='\t')

    for a in launch_log:
        u = a[0]
        d = int(a[1])
        if d in range(start, end + 1):
            user_embedding[u][d - start] += 1
        elif d in range(end + 1, end + 8):
            user_label[u] = 1

    for a in video_log:
        u = a[0]
        d = int(a[1])
        if d in range(start, end + 1):
            user_embedding[u][d - start + 7] += 1
        elif d in range(end + 1, end + 8):
            user_label[u] = 1

    for a in activity_log:
        u = a[0]
        d = int(a[1])
        if d in range(start, end + 1):
            user_embedding[u][d - start + 14] += 1
        elif d in range(end + 1, end + 8):
            user_label[u] = 1

    embeddings = []
    labels = []
    for u in sorted(user_embedding.keys()):
        embeddings.append(user_embedding[u])
        labels.append(user_label[u])

    np.savetxt('G:/dataset/a3d6_chusai_a_train/feature_1/user_embedding' + str(start) + '-' + str(end), embeddings, '%.2f')
    np.savetxt('G:/dataset/a3d6_chusai_a_train/feature_1/user_label' + str(start) + '-' + str(end), labels, '%d')


def generate_testdata():
    test_users = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/user24-30', dtype=str)

    user_embedding = dict()
    for u in test_users:
        user_embedding[u] = [0] * (7 * 3)
    launch_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/app_launch_log.txt',
                            dtype=str, delimiter='\t')
    video_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/video_create_log.txt',
                           dtype=str, delimiter='\t')
    activity_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_activity_log.txt',
                              dtype=str, delimiter='\t')

    for a in launch_log:
        u = a[0]
        if u not in test_users:
            continue
        d = int(a[1])
        if d in range(24, 31):
            user_embedding[u][d - 24] += 1

    for a in video_log:
        u = a[0]
        if u not in test_users:
            continue
        d = int(a[1])
        if d in range(24, 31):
            user_embedding[u][d - 24 + 7] += 1

    for a in activity_log:
        u = a[0]
        if u not in test_users:
            continue
        d = int(a[1])
        if d in range(24, 31):
            user_embedding[u][d - 24 + 14] + 1

    embeddings = []
    user_ids = []
    for u in sorted(user_embedding.keys()):
        embeddings.append(user_embedding[u])
        user_ids.append(u)

    np.savetxt('G:/dataset/a3d6_chusai_a_train/feature_1/test_user_embedding', embeddings, '%f')
    np.savetxt('G:/dataset/a3d6_chusai_a_train/feature_1/user_ids', user_ids, '%s')


def generate_train_users():
    train_users = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/feature_1/user17-23', dtype=str)
    ids = sorted(train_users)
    np.savetxt('G:/dataset/a3d6_chusai_a_train/feature_1/train_user_ids', ids, '%s')


def feature_extract(file_in, file_out):
    x = np.loadtxt(fname=file_in, dtype=float)
    u_feature = []
    for e in x:
        feature = []
        feature.extend(e[:7])
        freq = 0
        days = 0
        max_freq = max(e[:7])
        min_freq = min(e[:7])
        for i in range(7):
            if e[i] > 0:
                days += 1
            freq += e[i]
        avg_freq = freq / 7
        t = [(m - avg_freq) ** 2 for m in e[:7]]
        variance = sum(t) / 7
        feature.extend([freq, avg_freq, days, variance, max_freq, min_freq])

        feature.extend(e[7:14])
        freq = 0
        days = 0
        max_freq = max(e[7:14])
        min_freq = min(e[7:14])
        for i in range(7, 14):
            if e[i] > 0:
                days += 1
            freq += e[i]
        avg_freq = freq / 7
        t = [(m - avg_freq) ** 2 for m in e[7:14]]
        variance = sum(t) / 7
        feature.extend([freq, avg_freq, days, variance, max_freq, min_freq])

        feature.extend(e[14:])
        freq = 0
        days = 0
        max_freq = max(e[14:])
        min_freq = min(e[14:])
        for i in range(14, 21):
            if e[i] > 0:
                days += 1
            freq += e[i]
        avg_freq = freq / 7
        t = [(m - avg_freq) ** 2 for m in e[14:]]
        variance = sum(t) / 7
        feature.extend([freq, avg_freq, days, variance, max_freq, min_freq])

        u_feature.append(feature)

    np.savetxt(file_out, u_feature, '%.3f')


# get_user_between(17, 23)
# get_user_between(24, 30)
# generate_traindata()
# generate_testdata()

# generate_train_users()

get_user_between(10, 16)
generate_traindata(10, 16)
feature_extract('G:/dataset/a3d6_chusai_a_train/feature_1/user_embedding10-16',
                'G:/dataset/a3d6_chusai_a_train/feature_1/train_feature10-16')
# feature_extract('G:/dataset/a3d6_chusai_a_train/feature_1/test_user_embedding',
#                 'G:/dataset/a3d6_chusai_a_train/feature_1/test_feature')