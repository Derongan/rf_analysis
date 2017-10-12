import tensorflow as tf
from tensorflow import layers
from tensorflow import nn


class LSTMModel:
    """
    A object holding the model for training
    """

    def __init__(self, train_queue, **kwargs):
        """
        Initialize the model
        :param train_queue: The queue returning data as [batch,total_timesteps,2 (i and q)]
        :param classes: The number of classes to predict
        :param learning_rate: The learning rate of the model. Defaults to .001
        :param timestemps: Number of timesteps to run through. This will also determine the number of data points from the series we feed in at once
        :param bidirectional: Whether or not to use a bidirectional LSTM
        :param cell_type: Unused
        :param hidden: Number of LSTM hidden units. Defaults to 100
        """
        with tf.variable_scope("RF_LSTM"):
            # Set up model params
            self.num_classes = kwargs["classes"]  # How many classes are there
            self.max_length = kwargs["max_length"]  # The number of sequence values we use for input
            self.learning_rate = kwargs.get("learning_rate", .001)  # The learning rate for the model
            self.timesteps = kwargs["timesteps"]
            self.bidirectional = kwargs.get("bidirectional", False)

            # The number of samples we have to work with
            # must be divisible by the num we feed per recurrent step
            assert self.max_length % self.timesteps == 0

            self.inputs = tf.placeholder_with_default(train_queue[0], shape=(None, self.max_length, 2))
            self.labels = tf.placeholder_with_default(train_queue[1], shape=(None,))

            self.input_reshaped = tf.reshape(self.inputs,
                                             (-1, self.timesteps, int(2 * (self.max_length / self.timesteps))))

            self.train_one_hot = tf.one_hot(self.labels, self.num_classes)
            self.unstacked = tf.unstack(self.input_reshaped, self.timesteps, 1)

            lstm_cells = nn.rnn_cell.BasicLSTMCell(kwargs.get("hidden", 100), forget_bias=1.0, state_is_tuple=True)

            if not self.bidirectional:
                self.lstm_layer, _ = tf.nn.static_rnn(lstm_cells, self.unstacked, dtype=tf.float32)
            else:
                self.lstm_layer, _, _ = tf.nn.static_bidirectional_rnn(lstm_cells, lstm_cells, self.unstacked,
                                                                    dtype=tf.float32)

            lstm_last = self.lstm_layer[-1]

            output = layers.dense(lstm_last, self.num_classes)

            self.prediction = nn.softmax(output)

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.train_one_hot))
            self.train_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.prediction, axis=1), self.labels), dtype=tf.float32))

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self._setup_summaries()

    def _setup_summaries(self):
        loss_sum = tf.summary.scalar("Loss", self.loss)
        acc_sum = tf.summary.scalar("Accuracy", self.train_accuracy)

        self.summaries = tf.summary.merge([loss_sum, acc_sum])

    def train(self, session: tf.Session):
        """
        Train the model
        :param session: The tf.Session object to use
        :return: A list containing the resulting summaries and None
        """
        return session.run([self.summaries, self.loss, self.train_op])

    def test(self, session: tf.Session, inputs, labels):
        return session.run(self.summaries, feed_dict={self.inputs: inputs, self.labels: labels})

    def predict(self, session: tf.Session, samples):
        """
        Predict for a set of samples
        :param session: The tf.Session object to use
        :param samples: The samples to use as a numpy ndarray
        :return: The prediction
        """
        return session.run([self.prediction], feed_dict={self.inputs: samples})
