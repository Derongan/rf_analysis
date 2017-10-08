import argparse
import glob
import logging
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_example(fname, dfsample):
    device_num = int(re.search(r"\d+", fname).group())
    i_vals = dfsample["I"].values
    q_vals = dfsample["Q"].values

    example = tf.train.Example(features=tf.train.Features(feature={
        "device_num": _int64_feature(device_num),
        "i_values": _float_feature(i_vals),
        "q_values": _float_feature(q_vals)
    }))

    return example


def create_record(sample_files, path, i_min, i_max, q_min, q_max):
    """
    Takes a list of dataframes representing samples and saves a tfrecord to path
    :param sample_files:
    :param path:
    :return:
    """
    with open(path + ".tfrecord", "w") as fp:
        with tf.python_io.TFRecordWriter(fp.name) as writer:
            for sample_file in sample_files:
                sample = pd.read_csv(sample_file, sep=",", names=("I", "Q"))
                sample = (sample - (i_min, q_min)) / (i_max - i_min, q_max - q_min)
                example = create_example(os.path.basename(sample_file), sample)
                writer.write(example.SerializeToString())


def create_datset(in_root, out_root, shardsize=500, testsize=300):
    """
    Save the data into a number of tfrecord files. One set for testing one set for training. The data is normalized.
    This method assumes you have enough memory to read all data points.
    :param in_root: The root directory containing the raw CSV files
    :param out_root: The root directory for the outputted tfrecord files
    :param shardsize: The samples per tfrecord shard for the train set
    :param testsize: The number of samples in the test tfrecord file
    :return:
    """

    LOGGER.debug("Creating output paths")
    os.makedirs(os.path.join(out_root, "test"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "train"), exist_ok=True)

    LOGGER.info("Randomizing order")
    in_files = glob.glob(in_root + "/*.csv")
    np.random.shuffle(in_files)

    test_files = in_files[:testsize]
    train_file_shards = chunks(in_files[testsize:], shardsize)

    i_min = float("inf")
    q_min = float("inf")
    i_max = float("-inf")
    q_max = float("-inf")

    LOGGER.info("Finding minimum and maximum values for normalization")

    for num, file in enumerate(in_files):
        df = pd.read_csv(file, sep=",", names=("I", "Q"))
        temp_i_min, temp_q_min = np.min(df)
        temp_i_max, temp_q_max = np.max(df)

        i_min = min(i_min, temp_i_min)
        i_max = max(i_max, temp_i_max)

        q_min = min(q_min, temp_q_min)
        q_max = max(q_max, temp_q_max)

        if num % 100 == 0:
            LOGGER.debug("Finished reading sample {} for preprocessing".format(num))

    LOGGER.info("Saving test record")
    create_record(test_files, os.path.join(out_root, "test", "test_set"), i_min, i_max, q_min, q_max)

    LOGGER.info("Saving train shards")
    for i, shard in enumerate(train_file_shards):
        create_record(shard, os.path.join(out_root, "train", "train_set_{}".format(i)), i_min, i_max, q_min, q_max)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create tfrecord of rf data.')
    parser.add_argument('in_root', type=str)
    parser.add_argument('out_root', type=str)
    parser.add_argument('-s', '--shardsize', type=int, help="The number of examples per training record", default=500)
    parser.add_argument('-t', '--testsize', type=int, help="The number of examples in the test set", default=300)

    args = parser.parse_args()

    create_datset(**vars(args))
