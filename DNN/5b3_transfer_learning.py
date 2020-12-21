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

from cifarData import CIFAR10

MODEL_DIR = os.path.join(os.path.dirname(__file__),"models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "mnist_model.h5")

IMAGENET_SIZE = 96
IMAGENET_DEPTH = 3
IMAGENET_SHAPE = (IMAGENET_SIZE, IMAGENET_SIZE, IMAGENET_DEPTH)


def build_model(img_shape, num_classes) -> Model:
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=IMAGENET_SHAPE
    )

    num_layers = len(base_model.layers)
    print(f"Number of layers in the base model: {num_layers}")
    fine_tune_at = num_layers - 10
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    input_img = Input(shape=img_shape)
    x = Rescaling(scale=2.0, offset=-1.0)(input_img)
    x = Resizing(height=IMAGENET_SIZE, width=IMAGENET_SIZE)(x)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
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

    model = build_model(
        img_shape,
        num_classes
    )

    opt = Adam(learning_rate=5e-4)

    model.compile(
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

    model.fit(
        train_x,
        train_y,
        verbose=1,
        epochs=epochs,
        callbacks=[es_callback],
        validation_data=(val_x, val_y),
        batch_size=128
    )
    model.save_weights(filepath=MODEL_FILE_PATH)
    


    
    model.load_weights(filepath=MODEL_FILE_PATH)

    scores = model.evaluate(
    val_x,
    val_y,
    verbose=0
    )

    print(f"Scores: {scores}")

    nr_output_predictions = 6
    preds = model.predict(test_x[:nr_output_predictions])
    for i, pred in enumerate(preds):
        if i > nr_output_predictions :
            break
        pred_label, pred_probability = data.decode_labels(pred)
        y_label, y_probability = data.decode_labels(test_y[i])
        #plt.imshow(test_x[i])
        plt.imsave(f"output/transfer_learning/{i}_{pred_label}.jpg", test_x[i])
        
        print(f"Predicted class: {pred_label} ({pred_probability*100}%), Actual class: {y_label}")
        



# %%
