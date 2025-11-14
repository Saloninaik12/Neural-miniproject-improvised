import numpy as np
from tensorflow.keras.applications import (
    ResNet50, 
    VGG16, 
    MobileNetV2, 
    resnet50, 
    vgg16, 
    mobilenet_v2
)
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

# ---- Load available models ----
def load_model(model_name):
    if model_name == "ResNet50":
        return ResNet50(weights="imagenet")
    elif model_name == "VGG16":
        return VGG16(weights="imagenet")
    elif model_name == "MobileNetV2":
        return MobileNetV2(weights="imagenet")
    else:
        raise ValueError("Unknown model name!")

# ---- Preprocess image for selected model ----
def preprocess_for_model(img, model_name):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    if model_name == "ResNet50":
        arr = resnet50.preprocess_input(arr)
    elif model_name == "VGG16":
        arr = vgg16.preprocess_input(arr)
    elif model_name == "MobileNetV2":
        arr = mobilenet_v2.preprocess_input(arr)

    return arr

# ---- Decode predictions based on model ----
def decode_predictions(preds, model_name):
    if model_name == "ResNet50":
        return resnet50.decode_predictions(preds, top=5)[0]
    elif model_name == "VGG16":
        return vgg16.decode_predictions(preds, top=5)[0]
    elif model_name == "MobileNetV2":
        return mobilenet_v2.decode_predictions(preds, top=5)[0]

# ---- Grad-CAM Heatmap ----
def make_gradcam_heatmap(img_array, model, layer_name=None):
    if layer_name is None:
        layer_name = model.layers[-3].name  # pick last conv layer
    
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        pred_index = np.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = np.dot(conv_output, pooled_grads.numpy())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap
