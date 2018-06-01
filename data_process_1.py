import numpy as np


def generate_user_list(file):
    train_user = []
    test_user = []

    users = np.loadtxt(fname=file, dtype=str, delimiter='\t')

    for u in users:
        day = int(u[1])
        if day in range(1, 24):
            train_user.append(u[0])
        if day in range(8, 31):
            test_user.append(u[0])
    np.savetxt('G:/dataset/a3d6_chusai_a_train/user_register_log01-23', train_user, '%s')
    np.savetxt('G:/dataset/a3d6_chusai_a_train/user_register_log08-30', test_user, '%s')

# generate_user_list('G:/dataset/a3d6_chusai_a_train/user_register_log.txt')


def generate_traindata():
    train_users = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_register_log01-23', dtype=str)

    user_label = dict()
    user_embedding = dict()
    for u in train_users:
        user_label[u] = 0
        user_embedding[u] = [0] * (23 * 4)
    launch_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/app_launch_log.txt',
                            dtype=str, delimiter='\t')
    video_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/video_create_log.txt',
                           dtype=str, delimiter='\t')
    activity_log = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_activity_log.txt',
                              dtype=str, delimiter='\t')

    for a in launch_log:
        u = a[0]
        d = int(a[1])
        if d in range(1, 24):
            user_embedding[u][d - 1] += 1
        elif d in range(24, 31):
            user_label[u] = 1

    for a in video_log:
        u = a[0]
        d = int(a[1])
        if d in range(1, 24):
            user_embedding[u][d - 1 + 23] += 1
        elif d in range(24, 31):
            user_label[u] = 1

    for a in activity_log:
        u = a[0]
        d = int(a[1])
        author_id = a[4]
        action_type = int(a[5])
        if d in range(1, 24):
            user_embedding[u][d - 1 + 46] += 1
            # 根据用户的视频被播放、关注、 点赞、转发计算用户的活跃程度嵌入
            if author_id in user_embedding and action_type <= 4:
                user_embedding[author_id][d - 1 + 23 * 3] += action_type
        elif d in range(24, 31):
            user_label[u] = 1

    embeddings = []
    labels = []
    for u in sorted(user_embedding.keys()):
        embeddings.append(user_embedding[u])
        labels.append(user_label[u])

    np.savetxt('G:/dataset/a3d6_chusai_a_train/data4x23/user_embedding', embeddings, '%.2f')
    np.savetxt('G:/dataset/a3d6_chusai_a_train/data4x23/user_label', labels, '%d')


def generate_testdata():
    test_users = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_register_log08-30', dtype=str)

    user_embedding = dict()
    for u in test_users:
        user_embedding[u] = [0] * (23 * 4)
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
        if d in range(8, 31):
            user_embedding[u][d - 8] = 1

    for a in video_log:
        u = a[0]
        if u not in test_users:
            continue
        d = int(a[1])
        if d in range(8, 31):
            user_embedding[u][d - 8 + 23] = 1

    for a in activity_log:
        u = a[0]
        if u not in test_users:
            continue
        d = int(a[1])
        author_id = a[4]
        action_type = int(a[5])
        if d in range(8, 31):
            user_embedding[u][d - 8 + 46] = 1
            if author_id in user_embedding and action_type <= 4:
                user_embedding[author_id][d - 8 + 23 * 3] += action_type

    embeddings = []
    user_ids = []
    for u in sorted(user_embedding.keys()):
        embeddings.append(user_embedding[u])
        user_ids.append(u)

    np.savetxt('G:/dataset/a3d6_chusai_a_train/data4x23/test_user_embedding', embeddings, '%.2f')
    np.savetxt('G:/dataset/a3d6_chusai_a_train/data4x23/test_user_ids', user_ids, '%s')


def generate_train_users():
    train_users = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_register_log01-23', dtype=str)
    ids = sorted(train_users)
    np.savetxt('G:/dataset/a3d6_chusai_a_train/train_user_ids', ids, '%s')


# generate_traindata()
generate_testdata()
# generate_train_users()