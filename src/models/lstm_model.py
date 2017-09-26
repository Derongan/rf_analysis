import tensorflow as tf


def read_sequence_example(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'i_values': tf.FixedLenFeature([8000], tf.float32),
            'q_values': tf.FixedLenFeature([8000], tf.float32),
        })

    return features


def get_input():
    filename_queue = tf.train.string_input_producer(["../../data/processed/9_23_2017/9_23_2017.tfrecord"])

    return read_sequence_example(filename_queue)


init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as sess:
    thing = get_input()

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print(sess.run(thing))

    coord.request_stop()
    coord.join(threads)
