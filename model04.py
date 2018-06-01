import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


# 模型定义
def cnn_model_fn(features, labels, mode):
    # 输入层 4x23 ,通道1
    input_layer = tf.reshape(features['x'], [-1, 4, 23, 1])
    # 第1个卷积层 卷积核3x3，激活函数ReLU
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[4, 4],
        padding='SAME',
        activation=tf.nn.relu
    )

    # 第2个卷积层 卷积核4x4，激活函数ReLU
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=128,
        kernel_size=[4, 4],
        padding='SAME',
        activation=tf.nn.relu
    )
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=128,
        kernel_size=[4, 4],
        padding='VALID',
        activation=tf.nn.relu
    )
    # # 第2个汇合层 大小2x2
    # pool2 = tf.layers.max_pooling2d(
    #     inputs=conv2,
    #     pool_size=[2, 2],
    #     strides=1
    # )

    pool3_flat = tf.reshape(conv3, [-1, 1 * 20 * 128])
    # 全连接层FC1
    dense1 = tf.layers.dense(pool3_flat, units=1024, activation=tf.nn.relu)
    # dropout1
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 全连接层FC2
    dense2 = tf.layers.dense(dropout1, units=1024, activation=tf.nn.relu)
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

    eval_metrics_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions)}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


# 模型训练
def train():
    train_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/data4x23/user_embedding', dtype=float)
    train_y = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/data4x23/user_label', dtype=int)
    train_data = np.array(train_x, dtype=np.float32)
    train_labels = np.array(train_y, dtype=np.int32)

    cifar_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='G:/models/ksci_bdc_model/model04')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=50,
        num_epochs=None,
        shuffle=True)
    cifar_classifier.train(
        input_fn=train_input_fn,
        steps=100,
        # hooks=[logging_hook]
    )


# 预测
def predict():
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='G:/models/ksci_bdc_model/model04')

    test_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/data4x23/test_user_embedding', dtype=float)
    test_data = np.array(test_x, dtype=np.float32)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_data},
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    pre = [k for k in predictions]

    user_ids = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/data4x23/test_user_ids', dtype=str)
    active_users = []
    for i in range(len(pre)):
        if pre[i] == 1:
            active_users.append(user_ids[i])

    train_user_ids = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/train_user_ids', dtype=str)
    labels = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_label', dtype=int)
    for i in range(len(labels)):
        if labels[i] == 1 and train_user_ids[i] not in active_users:
            active_users.append(train_user_ids[i])

    np.savetxt('G:/dataset/a3d6_chusai_a_train/prediction/prediction04_2.txt', active_users, '%s')


def main(argv):
    # train()
    predict()


if __name__ == '__main__':
    tf.app.run()