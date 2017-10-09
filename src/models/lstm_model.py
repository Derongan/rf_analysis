import random
import glob

import numpy as np
import tensorflow as tf

LSTM_HIDDEN_UNITS = 50
NUM_CLASSES = 2
LEARNING_RATE = 1E-2
BATCH_SIZE = 4
NUM_EPOCHS = 100
TRAIN_SEQ_LEN = 2
SEQ_LEN = 8000
TRAIN_FILES = [glob.glob("../../data/processed/custom/train/*")]
TEST_FILE = "../../data/processed/custom/test/test_set.tfrecord"


def lstm(data, actual, name="my_lstm"):
    print(data.shape)
    data = tf.placeholder_with_default(data, [None, TRAIN_SEQ_LEN, 2], name="inputs")
    actual = tf.placeholder_with_default(actual, [None, NUM_CLASSES], name="labels")
    with tf.variable_scope(name):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_UNITS)

        # with tf.variable_scope("forwardLSTM"):
        # Forward LSTM
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)
        # We only care about the final state
        print(outputs.shape)
        last_output = outputs[:, -1, :]

        print(last_output.shape)

        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

        out_scores = tf.layers.dense(last_output, NUM_CLASSES, kernel_initializer=initializer,
                                                bias_initializer=initializer)

        print(out_scores.shape)
        print(actual.shape)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_scores, labels=actual))

        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        prediction = tf.nn.softmax(out_scores)

        test_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(actual, 1)), tf.float32))

        return train_op, test_op, loss, prediction, actual


def read_sequence_example(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'i_values': tf.FixedLenFeature([SEQ_LEN], tf.float32),
            'q_values': tf.FixedLenFeature([SEQ_LEN], tf.float32),
            "device_num": tf.FixedLenFeature([], tf.int64),
        })

    return features


def take_random_subsequence(features, size):
    """

    :param features: Data read in from a tfrecord
    :param size: the size of each cropping
    :return: i/q values cropped to size, label
    """

    iq = tf.stack([features["i_values"], features["q_values"]], axis=1)

    return iq[:size, :], features["device_num"]

    # ret_val = tf.random_crop(iq, [size, 2]), features["device_num"]
    # return ret_val


def get_input(seq_len):
    # Sadly the num_epochs argument is just flat broken? so we do this dumbness
    file_list = np.asarray([np.random.permutation(TRAIN_FILES) for _ in range(NUM_EPOCHS)])
    file_list = file_list.flatten().tolist()
    filename_queue = tf.train.string_input_producer(file_list)

    iq, device_num = take_random_subsequence(read_sequence_example(filename_queue), seq_len)

    # -1 cus we start our device nums at 1 right now
    device_one_hot = tf.one_hot(device_num - 1, NUM_CLASSES)

    data, labels = tf.train.shuffle_batch([iq, device_one_hot], batch_size=BATCH_SIZE, capacity=256,
                                          min_after_dequeue=10)

    return data, labels


def load_test_data():
    # tensorflow makes reading tfrecords really frigging annoying but hey, at least they work with queues
    i_values = []
    q_values = []
    device_num = []
    for x in tf.python_io.tf_record_iterator(TEST_FILE):
        example = tf.train.Example()
        example.ParseFromString(x)

        # hard coding cus I am not good at this
        partition = random.randint(0, SEQ_LEN - TRAIN_SEQ_LEN)

        i_values.append(example.features.feature["i_values"].float_list.value[partition:partition + TRAIN_SEQ_LEN])
        q_values.append(example.features.feature["q_values"].float_list.value[partition:partition + TRAIN_SEQ_LEN])
        device_num.append(example.features.feature["device_num"].int64_list.value[0])

    return [i_values, q_values], device_num


data, labels = get_input(TRAIN_SEQ_LEN)

test_data, test_labels = load_test_data()

test_data = np.asarray(test_data)
test_labels = np.asarray(test_labels)

# Voodoo magic one hot
# We subtract one because the current record stores starting at 1 not 0
n_values = NUM_CLASSES
test_labels = np.eye(n_values)[test_labels-1]

# make the axis pretty
test_data = np.transpose(test_data, [1, 2, 0])

train_op, test_op, loss, prediction_op, actual_op = lstm(data, labels)
init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = [tf.train.start_queue_runners(coord=coord) for i in range(16)]

    # Always keep track of this. Why not
    i = 0
    # We run until we have nothing left to read
    try:
        while True:
            # x = sess.run([data, labels])
            _, loss_val, train_accuracy, predictions, actual = sess.run([train_op, loss, test_op, prediction_op, actual_op])

            if (i % 10 == 0):
                print("Loss: {}, Train Accuracy: {}".format(loss_val, train_accuracy))

            if (i % 1000 == 0):
                print("Accuracy at step {}: ".format(i),
                      sess.run([test_op], feed_dict={"inputs:0": test_data, "labels:0": test_labels}))

            i += 1

    except tf.errors.OutOfRangeError:
        pass

    coord.request_stop()
    coord.join(threads)
