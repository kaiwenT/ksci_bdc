import tensorflow as tf
import numpy as np
import pandas as pd
tf.logging.set_verbosity(tf.logging.INFO)


# 模型定义
def cnn_model_fn(features, labels, mode):
    # 输入层 3x23 ,通道1
    input_layer = tf.reshape(features['x'], [-1, 111, 1])
    # input_layer = features['x']
    # 第1个卷积层 卷积核5x5，激活函数sigmoid
    conv1 = tf.layers.conv1d(
        inputs=input_layer,
        filters=64,
        kernel_size=5,
        padding='SAME',
        activation=tf.nn.tanh
    )

    # 第2个卷积层 卷积核5x5，激活函数ReLU
    conv2 = tf.layers.conv1d(
        inputs=conv1,
        filters=128,
        kernel_size=3,
        padding='SAME',
        activation=tf.nn.tanh
    )
    # 第2个汇合层 大小3x3
    pool2 = tf.layers.max_pooling1d(
        inputs=conv2,
        pool_size=3,
        strides=1
    )

    pool3_flat = tf.reshape(pool2, [-1, 1 * 109 * 128])
    # 全连接层FC1
    dense1 = tf.layers.dense(pool3_flat, units=128, activation=tf.nn.tanh)
    # dropout1
    # dropout1 = tf.layers.dropout(
    #     inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 全连接层FC2
    # dense2 = tf.layers.dense(dense1, units=1024, activation=tf.nn.tanh)
    # dropout2
    # dropout2 = tf.layers.dropout(
    #     inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 输出层
    logits = tf.layers.dense(inputs=dense1, units=2)

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
    train_xy = pd.read_csv('G:/dataset/a3d6_chusai_a_train/dealt_data/train_test/train.csv')
    # print(train_xy.columns)
    train_data = train_xy.drop(['user_id', 'label'], axis=1)
    train_xy['label'].fillna(1, inplace=True)
    train_labels = train_xy['label']
    print(len(train_data.columns))
    train_data = np.array(train_data, dtype=np.float32)

    train_labels = np.array(train_labels, dtype=np.int32)

    cnn_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='G:/models/ksci_bdc_model/feature111_cnn')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=50,
        num_epochs=1,
        shuffle=False)
    cnn_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        # hooks=[logging_hook]
    )

    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={'x': train_data},
    #     y=train_labels,
    #     num_epochs=1,
    #     shuffle=False)
    # ret = cnn_classifier.evaluate(input_fn=eval_input_fn)
    # print(ret)
    # f1_score = 2 * ret['precision'] * ret['recall'] / (ret['precision'] + ret['recall'])
    # print('F1-Score: ', f1_score)
    # return f1_score


# 模型评估
def evaluate():
    eval_xy = pd.read_csv('G:/dataset/a3d6_chusai_a_train/dealt_data/train_test/validate.csv')
    # print(train_xy.columns)
    eval_data = eval_xy.drop(['user_id', 'label'], axis=1)
    eval_xy['label'].fillna(1, inplace=True)
    eval_labels = eval_xy['label']
    print(len(eval_data.columns))
    eval_data = np.array(eval_data, dtype=np.float32)

    eval_labels = np.array(eval_labels, dtype=np.int32)

    cnn_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='G:/models/ksci_bdc_model/feature111_cnn')

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    ret = cnn_classifier.evaluate(input_fn=eval_input_fn)
    print(ret)
    f1_score = 2 * ret['precision'] * ret['recall'] / (ret['precision'] + ret['recall'])
    print('F1-Score: ', f1_score)
    return f1_score


# 预测
def predict(fname):
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='G:/models/ksci_bdc_model/feature111_cnn')

    test_x = pd.read_csv('G:/dataset/a3d6_chusai_a_train/dealt_data/train_test/test.csv')
    test_data = test_x.drop(['user_id'], axis=1)
    test_data = np.array(test_data, dtype=np.float32)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_data},
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    pre = [k for k in predictions]

    pred = pd.DataFrame()
    pred['user_id'] = test_x['user_id']
    pred['active'] = pre
    active_users = pred[(pred['active'] > 0)]['user_id']

    active_users.to_csv('G:/dataset/a3d6_chusai_a_train/prediction/predict_feature111cnn_' + fname + '.txt',
                        index=False, header=None)


def main(argv):
    # train()
    # predict1('0.80')
    # evaluate()
    train()
    score = evaluate()
    # while score < 0.82:
    #     score = train()
    predict(str(score))


if __name__ == '__main__':
    tf.app.run()