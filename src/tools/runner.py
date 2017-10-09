import tensorflow as tf
from models.smarter_model import LSTMModel
from data_loader import DataLoader

NUM_CLASSES = 2
INPUT_LEN = 500

if __name__ == "__main__":
    loader = DataLoader("../9_23_2017", 100, crop_size=INPUT_LEN)
    model = LSTMModel(loader.get_samples(), classes=2, length=INPUT_LEN, learning_rate=.01)
    with tf.Session() as sess:
        pass
