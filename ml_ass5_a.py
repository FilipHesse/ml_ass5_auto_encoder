import os

import numpy as np
import matplotlib.pyplot as plt
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

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def plot_encoded(encoded_imgs, y_test, class0, class1):
    fig, ax = plt.subplots()
    for i, img in enumerate(encoded_imgs):
        marker = ''
        if y_test[i] == 0:
            marker = 'r+'
            label_str = f"digit {class0}"
        if y_test[i] == 1:
            marker = 'b+'
            label_str = f"digit {class1}"
        ax.plot(img[0], img[1], marker, label = label_str)
    legend_without_duplicate_labels(ax)
    plt.xlabel('a_0')
    plt.ylabel('a_1')
    plt.title(f'Output of hidden layer: {class0} vs {class1}')
    plt.savefig(f'logs/encoded_{class0}_vs_{class1}.png')
    

def plot_decoded(decoded_imgs, class0, class1):
    n = 5 
    perm = np.random.permutation(np.shape(decoded_imgs)[0])
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[perm[i]].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[perm[i]].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(f'logs/decoded_{class0}_vs_{class1}_subs1.png')

    #Plot other subset
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[perm[i+n]].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[perm[i]+n].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(f'logs/decoded_{class0}_vs_{class1}_subs2.png')


if __name__ == "__main__":
    num_features = 784
    num_classes = 784
    class0 = 0
    class1 = 1

    (x_train, y_train), (x_test, y_test) = prepare_dataset_autoencoder(
        num_features, class0, class1)

    optimizer = Adam()
    epochs = 30
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

    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    plot_encoded(encoded_imgs, y_test, class0, class1)

    plot_decoded(decoded_imgs, class0, class1)

