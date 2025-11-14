import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    resnet50, vgg16, mobilenet
)
from tensorflow.keras.preprocessing import image

# -------------------- LOAD MODEL --------------------
def load_model(name):
    if name == "ResNet50":
        model = resnet50.ResNet50(weights="imagenet")
        preprocess = resnet50.preprocess_input
        decode = resnet50.decode_predictions

    elif name == "VGG16":
        model = vgg16.VGG16(weights="imagenet")
        preprocess = vgg16.preprocess_input
        decode = vgg16.decode_predictions

    elif name == "MobileNet":
        model = mobilenet.MobileNet(weights="imagenet")
        preprocess = mobilenet.preprocess_input
        decode = mobilenet.decode_predictions

    return model, preprocess, decode


# -------------------- PREPROCESS FUNCTION --------------------
def preprocess_for_model(img, preprocess_fn):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_fn(arr)
    return arr


# -------------------- GRAD-CAM --------------------
def make_gradcam_heatmap(img_array, model, last_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        pred_output = preds[:, pred_index]

    grads = tape.gradient(pred_output, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    return heatmap.numpy()
