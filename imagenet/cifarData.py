from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


np.random.seed(0)
tf.random.set_seed(0)


class CIFAR10:
    def __init__(self, with_normalization: bool = True, validation_size: float = 0.33) -> None:
        # User-definen constants
        self.num_classes = 10
        # Load the data set
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Split the dataset
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)
        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)
        if with_normalization:
            self.x_train = self.x_train / 255.0
        if with_normalization:
            self.x_test = self.x_test / 255.0
        if with_normalization:
            self.x_val = self.x_val / 255.0
        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes)
        self.y_val = to_categorical(y_val, num_classes=self.num_classes)
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.val_size = self.x_val.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_val_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_val, self.y_val

    def load_label_names(self):
        return np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    def decode_labels(self, y_numeric):
        max_probability = np.amax(y_numeric)
        max_label = self.load_label_names()[y_numeric==max_probability][0]
        return max_label, max_probability


