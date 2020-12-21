#%%
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import os
import matplotlib.pyplot as plt

import pickle

from cifarData_mobileNet import CIFAR10
from cifarData_mobileNet import CIFAR10

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
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=CIFAR_SHAPE #(224, 224, 3)
    )
    #features = base_model.layers[-1].output

    num_layers = len(base_model.layers)
    print(f"Number of layers in the base model: {num_layers}")

    for layer in base_model.layers:
        layer.trainable = False

        if 'conv' not in layer.name:
            pass

        filters = layer.get_weights()
        for filt in filters:
            print(layer.name, filt.shape)

    input_img = Input(shape=img_shape)
    features = base_model(input_img)
    #x = GlobalAveragePooling2D()(x)
    #x = Dense(units=num_classes)(x)
    #y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
        outputs=[features]
    )

    model.summary()

    return model

def build_my_model(feature_shape, num_classes) -> Model:

    num_layers = len(base_model.layers)
    print(f"Number of layers in the my model: {num_layers}")

    for layer in base_model.layers:
        layer.trainable = False

    input_features = Input(shape=feature_shape)
    x = GlobalAveragePooling2D()(input_features)
    x = Dense(units=1000)(x)
    x = Activation("relu")(x)
    x = Dense(units=500)(x)
    x = Activation("relu")(x)
    x = Dense(units=250)(x)
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

    img_shape = data.img_shape
    num_classes = data.num_classes


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

    # base_model.fit(
    #     train_x,
    #     train_y,
    #     verbose=1,
    #     epochs=epochs,
    #     callbacks=[es_callback],
    #     validation_data=(val_x, val_y),
    #     batch_size=128
    # )



    
    # take a subset of trainingset, because GPU cant handle entire trainingset
    features_train = base_model.predict(train_x, batch_size=32)
    features_val = base_model.predict(val_x, batch_size=32)

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
    # for i, pred in enumerate(preds):
    #     if i > nr_output_predictions :
    #         break
    #     pred_label, pred_probability = data.decode_labels(pred)
    #     y_label, y_probability = data.decode_labels(test_y[i])
    #     #plt.imshow(test_x[i])
    #     plt.imsave(f"output/transfer_learning/{i}_{pred_label}.jpg", test_x[i])
        
    #     print(f"Predicted class: {pred_label} ({pred_probability*100}%), Actual class: {y_label}")
    
    # model.save_weights(filepath=MODEL_FILE_PATH)

    # model.load_weights(filepath=MODEL_FILE_PATH)



# %%
