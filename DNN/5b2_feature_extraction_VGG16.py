#%%
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import os
import matplotlib.pyplot as plt

import pickle

from cifarData import CIFAR10

MODEL_DIR = os.path.join(os.path.dirname(__file__),"models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "feature_extraction_model.h5")

IMAGENET_SIZE = 96
IMAGENET_DEPTH = 3
IMAGENET_SHAPE = (IMAGENET_SIZE, IMAGENET_SIZE, IMAGENET_DEPTH)

CIFAR_SIZE = 32
CIFAR_DEPTH = 3
CIFAR_SHAPE = (CIFAR_SIZE, CIFAR_SIZE, CIFAR_DEPTH)


def build_base_model(img_shape, num_classes) -> Model:
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=CIFAR_SHAPE #(224, 224, 3)
    )
    #features = base_model.layers[-1].output

    num_layers = len(base_model.layers)
    print(f"Number of layers in the base model: {num_layers}")

    for i, layer in enumerate(base_model.layers):
        layer.trainable = False


        filters = layer.get_weights()
        for filt in filters:
            print(i, layer.name, filt.shape)

    input_img = Input(shape=img_shape)
    features = base_model(input_img)
    #x = GlobalAveragePooling2D()(x)
    #x = Dense(units=num_classes)(x)
    #y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[base_model.input],
        #outputs=[base_model.output]
        outputs=[base_model.get_layer('block3_pool').output]
    )

    model.summary()

    return model

def build_my_model(feature_shape, num_classes) -> Model:

    num_layers = len(base_model.layers)
    print(f"Number of layers in the my model: {num_layers}")

    for layer in base_model.layers:
        layer.trainable = False

    input_features = Input(shape=feature_shape)
    #x = MaxPooling2D()(input_features)
    x = Conv2D(512, 3, padding="same")(input_features)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(1024, 3, padding="same")(input_features)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=512)(x)
    x = Activation("relu")(x)
    x = Dense(units=256)(x)
    x = Activation("relu")(x)
    x = Dense(units=128)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_features],
        outputs=[y_pred]
    )

    model.summary()

    return model


if __name__ == "__main__":
    data = CIFAR10()

    (train_x, train_y) = data.get_train_set()
    (val_x, val_y) = data.get_val_set()
    (test_x, test_y) = data.get_test_set()

    plt.imsave(f"output/feature_extraction/test_image_0.jpg", train_x[0])

    img_shape = data.img_shape
    num_classes = data.num_classes

    train_x = preprocess_input(train_x)
    val_x = preprocess_input(val_x)
    test_x = preprocess_input(test_x)

    # Global params
    epochs = 100

    base_model = build_base_model(
        img_shape,
        num_classes
    )

    opt = Adam(learning_rate=5e-4)

    base_model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    es_callback = EarlyStopping(
        monitor="val_loss",
        patience=30,
        verbose=1,
        restore_best_weights=True
    )

    features_train = base_model.predict(train_x, batch_size=32)
    features_val = base_model.predict(val_x, batch_size=32)

#%%
    square = 8
    i=1
    fig = plt.figure()
    for _ in range(square):
        for _ in range(square):
            ax = fig.add_subplot(square, square, i)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(features_train[0,:,:,i-1],cmap='gray')
            i += 1

    fig.savefig(f"output/feature_extraction/test_image_0_features.jpg")

    my_model = build_my_model(
        features_train.shape[1:],
        num_classes
    )

    my_model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    my_model.fit(
        features_train,
        train_y,
        verbose=1,
        epochs=epochs,
        callbacks=[es_callback],
        validation_data=(features_val
        , val_y),
        batch_size=128
    )

    scores = my_model.evaluate(
        features_val,
        val_y,
        verbose=0
    )

    print(f"Scores: {scores}")

# %%
