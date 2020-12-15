# import the necessary packages
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_RN152V2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet152V2
import numpy as np
import os
import cv2
# construct the argument parser and parse the arguments

def classify_image(image_path):

    # load the original image via OpenCV so we can draw on it and display
    # it to our screen later
    orig = cv2.imread(image_path)

    # load the input image using the Keras helper utility while ensuring
    # that the image is resized to 224x224 pxiels, the required input
    # dimensions for the network -- then convert the PIL image to a
    # NumPy array
    print("[INFO] loading and preprocessing image...")
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)

    # our image is now represented by a NumPy array of shape (224, 224, 3),
    # assuming TensorFlow "channels last" ordering of course, but we need
    # to expand the dimensions to be (1, 3, 224, 224) so we can pass it
    # through the network -- we'll also preprocess the image by subtracting
    # the mean RGB pixel intensity from the ImageNet dataset
    image = np.expand_dims(image, axis=0)

    # load the VGG16 network pre-trained on the ImageNet dataset
    print("[INFO] loading network...")
    #image = preprocess_vgg16(image)
    #model = VGG16(weights="imagenet")

    image = preprocess_RN152V2(image)
    model = ResNet152V2(weights="imagenet")

    # classify the image
    print("[INFO] classifying image...")
    preds = model.predict(image)
    P = decode_predictions(preds)

    # loop over the predictions and display the rank-5 predictions +
    # probabilities to our terminal
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1,
         label, prob * 100))

    # load the image via OpenCV, draw the top prediction on the image,
    # and display the image to our screen
    orig = cv2.imread(image_path)
    (imagenetID, label, prob) = P[0][0]
    cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Classification", orig)
    cv2.waitKey(0)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    path_to_images = os.path.join(os.path.dirname(__file__),"test_images")
    #images = ["bottle1.jpg","bottle2.jpg", "pen.jpg", "phone.jpg", "shoe.jpg","apple.jpg", "beerbottle.jpg", "winebottle.jpg", "broccoli.jpg","mouse.jpg"]
    images = ["mouse.jpg"]
    for im in images:
        image_fullpath = os.path.join(path_to_images, im)
        classify_image(image_fullpath)