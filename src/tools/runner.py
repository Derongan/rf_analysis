import tensorflow as tf

from data_loader import DataLoader
from models.LSTMModel import LSTMModel

DATA_DIR = "../../data/processed/9_23_2017/train/"
TEST_DATA_DIR = "../../data/processed/9_23_2017/test/"
LOG_DIR = "../../logs/9_23_2017"
NUM_CLASSES = 2
INPUT_LEN = 6000
DOWNSAMPLE = 4
MODEL_IN_LEN = 6000 / DOWNSAMPLE
TEST_COUNT = 100

if __name__ == "__main__":
    trainLoader = DataLoader(DATA_DIR, 100, crop_size=INPUT_LEN, randomize_crop=True, batch_size=32, file_limit=1000,
                             down_sample_step=DOWNSAMPLE)
    testLoader = DataLoader(TEST_DATA_DIR, 1, crop_size=INPUT_LEN, randomize_crop=False, batch_size=TEST_COUNT,
                            file_limit=1, down_sample_step=DOWNSAMPLE, noise=False)
    model = LSTMModel(trainLoader.get_samples(), classes=2, max_length=MODEL_IN_LEN, learning_rate=.001, hidden=100,
                      timesteps=50)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph, flush_secs=5)
        test_writer = tf.summary.FileWriter(LOG_DIR + '/test', flush_secs=5)

        init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = [tf.train.start_queue_runners(coord=coord) for i in range(16)]

        # TODO this is still hacky. We grab the first n samples from the test set as our validation
        test_inputs, test_labels = sess.run(testLoader.get_samples())

        for i in range(100000):
            summary, loss, _ = model.train(sess)
            train_writer.add_summary(summary, i)

            if i % 100 == 0:
                summary = model.test(sess, test_inputs, test_labels)
                test_writer.add_summary(summary, i)
