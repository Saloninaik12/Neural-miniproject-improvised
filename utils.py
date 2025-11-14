import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    resnet50, vgg16, mobilenet
)
from tensorflow.keras.preprocessing import image

# -----------------------
# Model Loader
# -----------------------
def load_model(model_name):
    if model_name == "ResNet50":
        model = resnet50.ResNet50(weights='imagenet')
        preprocess = resnet50.preprocess_input
        decode = resnet50.decode_predictions

    elif model_name == "VGG16":
        model = vgg16.VGG16(weights='imagenet')
        preprocess = vgg16.preprocess_input
        decode = vgg16.decode_predictions

    elif model_name == "MobileNet":
        model = mobilenet.MobileNet(weights='imagenet')
        preprocess = mobilenet.preprocess_input
        decode = mobilenet.decode_predictions

    else:
        raise ValueError("Unknown model selected")

    return model, preprocess, decode


# -----------------------
# Preprocess Image
# -----------------------
def preprocess_for_model(img, preprocess_fn):
    img = img.resize((224, 224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_fn(img_arr)
    return img_arr


# -----------------------
# GradCAM (No OpenCV)
# Pure TensorFlow Implementation
# -----------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_index]

        grads = tape.gradient(pred_output, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(
        tf.multiply(pooled_grads, conv_outputs), axis=-1
    )

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    return heatmap
