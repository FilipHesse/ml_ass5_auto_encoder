import os

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from tensorflow import keras
from tensorflow.keras import layers

from tf_utils.callbacks import ConfusionMatrix


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "mnist_autoencoder.h5")
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_autoencoder")


def prepare_dataset(num_features: int, num_classes: int) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)

    return (x_train, y_train), (x_test, y_test)

def prepare_dataset_autoencoder(num_features: int, class0: int, class1:int ) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    mask_train_0_1 = np.logical_or(y_train == class0 ,y_train == class1)
    mask_test_0_1 = np.logical_or(y_test == class0 ,y_test == class1)

    # reshape, as float32, normalize, apply mask
    x_train = (x_train.reshape(-1, num_features).astype(np.float32)/255)[mask_train_0_1]
    x_test = (x_test.reshape(-1, num_features).astype(np.float32)/255)[mask_test_0_1]

    y_train = y_train[mask_train_0_1]
    y_test = y_test[mask_test_0_1]

    y_train[y_train == class0] = 0
    y_train[y_train == class1] = 1

    y_test[y_test == class0] = 0
    y_test[y_test == class1] = 1

    return (x_train, y_train), (x_test, y_test)


def build_models(num_features: int, num_classes: int) -> Sequential:
    encoding_dim = 2

    # This is our input image
    input_img = keras.Input(shape=(num_features,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(num_classes, activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)

    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))


    return autoencoder, encoder, decoder


if __name__ == "__main__":
    num_features = 784
    num_classes = 784

    (x_train, y_train), (x_test, y_test) = prepare_dataset_autoencoder(num_features, class0=2, class1=3)

    optimizer = Adam()
    epochs = 100
    batch_size = 256

    autoencoder, encoder, decoder = build_models(num_features, num_classes)

    autoencoder.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        histogram_freq=1,
        write_graph=True
    )

    classes_list = [class_idx for class_idx in range(num_classes)]

    autoencoder.fit(
        x=x_train,
        y=x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[tb_callback],
    )

    scores = autoencoder.evaluate(
        x=x_test,
        y=x_test,
        verbose=0
    )
    print("Scores: ", scores)
