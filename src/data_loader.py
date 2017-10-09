import logging
import os

import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class DataLoader:
    """
    Handles the input pipeline for models
    """

    def __init__(self, record_dir, epochs, num_threads=8, full_seq_len=8000, crop_size=500,
                 randomize_crop=True,
                 noise=None):
        """
        Initialize the pipeline
        :param record_dir: The record containing all training tfrecords
        :param epochs: The number of epochs to prepare
        :param num_threads: The number of parallel threads to to use
        :param full_seq_len: The number of data points per sample
        """

        self.full_seq_len = full_seq_len
        self.crop_size = crop_size
        self.noise = noise

        search_glob = os.path.join(record_dir, "*.tfrecord")
        matched_files = tf.train.match_filenames_once(search_glob)

        try:
            self.filename_queue = tf.train.string_input_producer(matched_files, epochs)
        except tf.errors.FailedPreconditionError:
            LOGGER.error("Got precondition error, you probably forgot to initialize local variables")
            exit(1)

        self._sample_read()
        self._preprocess_samples(randomize_crop)

    def _sample_read(self):
        """
        Process the queue into sample tensors
        :return:
        """
        try:
            _, serialized_example = self.reader.read(self.filename_queue)
        except AttributeError:
            LOGGER.debug("Initializing record reader")
            self.reader = tf.TFRecordReader()
            _, serialized_example = self.reader.read(self.filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'i_values': tf.FixedLenFeature([self.full_seq_len], tf.float32),
                'q_values': tf.FixedLenFeature([self.full_seq_len], tf.float32),
                "device_num": tf.FixedLenFeature([], tf.int64),
            })

        self.parsed_samples = features

    def _preprocess_samples(self, is_random):
        """
        Perform preprocessing on the samples
        :return:
        """
        iq = tf.stack([self.parsed_samples["i_values"], self.parsed_samples["q_values"]], axis=1)

        if is_random:
            self.data = tf.random_crop(iq, self.crop_size)
        else:
            self.data = iq[:self.crop_size, :]

        if self.noise:
            self.data += tf.random_normal(iq.shape)

        # TODO this is a bit hacky based on how we read data. We should be able to have nice one hots regardless of what we name our samples
        self.labels = self.parsed_samples["device_num"] - 1

    def get_samples(self):
        return self.data, self.labels
