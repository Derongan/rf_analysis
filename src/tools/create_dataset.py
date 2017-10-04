import os

import random
import tensorflow as tf


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_list_feature(value):
    # Same as _float_feature but assumes value is a list already
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# yea this is ugly, sue me
device_dict = {}
cur_device = [0]

def create_sequence_example(file_name):
    # Yes most of this could be done nicer using a csv parser but I'd rather just get it done
    with open(file_name) as file:
        file_string = os.path.basename(file_name)
        device = file_string.split("-")[0]
        sample_num = int(file_string.split("-")[1][:-8].strip())

        sample_length = 0

        i_values = []
        q_values = []

        if device not in device_dict:
            device_dict[device] = cur_device[0]
            cur_device[0] += 1

        device_num = device_dict[device]

        for line in file:
            i, q = line.split(',')
            i_values.append(float(i))
            q_values.append(float(q))
            sample_length += 1

        example = tf.train.Example(features=tf.train.Features(feature={
            "device_name": _bytes_feature(device.encode()),
            "device_num": _int64_feature(device_num),
            "sample_num": _int64_feature(sample_num),
            "length": _int64_feature(sample_length),
            "i_values": _float_list_feature(i_values),
            "q_values": _float_list_feature(q_values)
        }))

    return example


def create_tf_record(in_dir, out_record_name, hold_out_percent=.35):
    files = os.listdir(in_dir)
    # filter data
    files = [file for file in files if file.endswith("Samp.csv") and os.path.isfile(os.path.join(in_dir, file))]

    num_files = len(files)
    division_index = int(num_files * hold_out_percent)
    # in place shuffle
    random.shuffle(files)

    train_set = files[division_index:]
    test_set = files[:division_index]

    with open(out_record_name + "_train.tfrecord", "w") as fp:
        with tf.python_io.TFRecordWriter(fp.name) as writer:
            for file in train_set:
                example = create_sequence_example(os.path.join(in_dir, file))
                writer.write(example.SerializeToString())

    with open(out_record_name + "_test.tfrecord", "w") as fp:
        with tf.python_io.TFRecordWriter(fp.name) as writer:
            for file in test_set:
                example = create_sequence_example(os.path.join(in_dir, file))
                writer.write(example.SerializeToString())


if __name__ == "__main__":
    create_tf_record("../../data/raw/test", "../../data/processed/test/test")
    print(len(device_dict))
