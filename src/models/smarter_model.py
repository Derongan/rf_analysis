import tensorflow as tf
from tensorflow import layers
from tensorflow import nn


class LSTMModel:
    """
    A object holding the model for training
    """
    def __init__(self, train_queue, **kwargs):
        with tf.variable_scope("RF_LSTM"):
            # Set up model params
            self.num_classes = kwargs["classes"]  # How many classes are there
            self.max_length = kwargs["length"]  # The number of sequence values we use for input
            self.learning_rate = kwargs["learning_rate"]  # The learning rate for the model

            self.inputs = tf.placeholder_with_default(train_queue[0], shape=(None, self.max_length, 2))
            self.train_labels = tf.placeholder_with_default(train_queue[1], shape=(None, self.num_classes))

            lstm_cells = nn.rnn_cell.BasicLSTMCell(kwargs["hidden"])

            lstm_layer = tf.nn.dynamic_rnn(lstm_cells, self.inputs, dtype=tf.float32)

            output = layers.dense(lstm_layer, self.num_classes)

            self.prediction = nn.softmax(output)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.train_labels))
            self.train_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.prediction), tf.argmax(self.learning_rate))))

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            tf.summary.histogram("Loss", self.loss)
            tf.summary.histogram("Accuracy", self.train_accuracy)

    def train(self, session, ):
        """
        Train the model
        :param session: The tf.Session object to use
        :return:
        """

    def predict(self, session: tf.Session, samples):
        """
        Predict for a set of samples
        :param session: The tf.Session object to use
        :param samples: The samples to use as a numpy ndarray
        :return: The prediction
        """
        return session.run([self.prediction], feed_dict={self.inputs: samples})
