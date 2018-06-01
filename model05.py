import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


# 模型定义
def cnn_model_fn(features, labels, mode):
    # 输入层 3x23 ,通道1
    input_layer = tf.reshape(features['x'], [-1, 3, 23, 1])
    # 第1个卷积层 卷积核5x5，激活函数sigmoid
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.tanh
    )

    # 第2个卷积层 卷积核5x5，激活函数ReLU
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=128,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.tanh
    )
    # 第2个汇合层 大小3x3
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[3, 3],
        strides=1
    )

    pool3_flat = tf.reshape(pool2, [-1, 1 * 21 * 128])
    # 全连接层FC1
    dense1 = tf.layers.dense(pool3_flat, units=1024, activation=tf.nn.tanh)
    # dropout1
    # dropout1 = tf.layers.dropout(
    #     inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 全连接层FC2
    dense2 = tf.layers.dense(dense1, units=1024, activation=tf.nn.tanh)
    # dropout2
    # dropout2 = tf.layers.dropout(
    #     inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 输出层
    logits = tf.layers.dense(inputs=dense2, units=2)

    # 预测结果
    predictions = tf.argmax(input=logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    precision = tf.metrics.precision(
        labels=labels, predictions=predictions)
    recall = tf.metrics.recall(
        labels=labels, predictions=predictions)
    eval_metrics_ops = {
        'precision': precision,
        'recall': recall,
        # 'f1-score': 2 * precision * recall / (precision + recall)
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


# 模型训练
def train():
    train_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_embedding', dtype=float)
    train_y = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_label', dtype=int)
    train_data = np.array(train_x, dtype=np.float32)
    train_labels = np.array(train_y, dtype=np.int32)

    cifar_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='G:/models/ksci_bdc_model/model05')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=50,
        num_epochs=None,
        shuffle=True)
    cifar_classifier.train(
        input_fn=train_input_fn,
        steps=2,
        # hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)
    ret = cifar_classifier.evaluate(input_fn=eval_input_fn)
    print(ret)
    f1_score = 2 * ret['precision'] * ret['recall'] / (ret['precision'] + ret['recall'])
    print('F1-Score: ', f1_score)
    return f1_score


# 模型评估
def evaluate():
    train_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_embedding', dtype=float)
    train_y = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_label', dtype=int)
    train_data = np.array(train_x, dtype=np.float32)
    train_labels = np.array(train_y, dtype=np.int32)

    cifar_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='G:/models/ksci_bdc_model/model05')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)
    ret = cifar_classifier.evaluate(input_fn=train_input_fn)
    print(ret)
    f1_score = 2 * ret['precision'] * ret['recall'] / (ret['precision'] + ret['recall'])
    print('F1-Score: ', f1_score)


# 预测
def predict1(fname):
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='G:/models/ksci_bdc_model/model05')

    test_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/test_user_embedding', dtype=float)
    test_data = np.array(test_x, dtype=np.float32)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_data},
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    pre = [k for k in predictions]

    user_ids = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_ids', dtype=str)
    active_users = []
    for i in range(len(pre)):
        if pre[i] == 1:
            active_users.append(user_ids[i])

    train_user_ids = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/train_user_ids', dtype=str)
    labels = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_label', dtype=int)
    for i in range(len(labels)):
        if labels[i] == 1 and train_user_ids[i] not in active_users:
            active_users.append(train_user_ids[i])

    np.savetxt('G:/dataset/a3d6_chusai_a_train/prediction/prediction05_' + fname + '.txt', active_users, '%s')


def main(argv):
    # train()
    # predict1('0.80')
    # evaluate()
    score = train()
    while score < 0.82:
        score = train()
    predict1(str(score))


if __name__ == '__main__':
    tf.app.run()