import numpy as np
import tensorflow as tf
from sklearn import model_selection

tf.logging.set_verbosity(tf.logging.INFO)


def train(train_data, test_data, train_label, test_label):
    validation_metrics = {
        "accuracy": tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key=tf.contrib.learn.PredictionKey.
            CLASSES),
        "precision": tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key=tf.contrib.learn.PredictionKey.
            CLASSES),
        "recall": tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key=tf.contrib.learn.PredictionKey.
            CLASSES)
    }

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        test_data,
        test_label,
        every_n_steps=50,
        metrics=validation_metrics,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=800)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=23 * 3)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[20, 40, 40, 20],
                                                n_classes=2,
                                                model_dir="G:/models/ksci_bdc_model/model_dnn",
                                                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

    # Fit model.
    classifier.fit(x=train_data,
                   y=train_label,
                   steps=5000,
                   monitors=[validation_monitor])

    # Evaluate accuracy.
    eva = classifier.evaluate(x=test_data, y=test_label)
    print(eva)
    accuracy_score = eva["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))
    # print('precision: ', precision_score)
    # print('recall: ', recall_score)
    # print('F1-Score: ', 2 * precision_score * recall_score / (precision_score + recall_score))


def predict(test_data):
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=23 * 3)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[20, 40, 40, 20],
                                                n_classes=2,
                                                model_dir="G:/models/ksci_bdc_model/model_dnn",
                                                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
    pred = classifier.predict(test_data)
    pre = [round(x) for x in pred]

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

    np.savetxt('G:/dataset/a3d6_chusai_a_train/prediction/prediction_' + 'dnn_4layers' + '.txt', active_users, '%s')


def main(unused_argv):
    train_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_embedding', dtype=float)
    train_y = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/user_label', dtype=int)
    train_data, test_data, train_label, test_label = model_selection.train_test_split(
        train_x, train_y, test_size=0.1, random_state=0)
    train(train_data, test_data, train_label, test_label)

    test_x = np.loadtxt(fname='G:/dataset/a3d6_chusai_a_train/test_user_embedding', dtype=float)
    predict(test_x)


if __name__ == "__main__":
  tf.app.run()