################################################################################
#MY CODE
################################################################################

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

# -----------------------------
# Load feature extractor (DenseNet201 without top)
# -----------------------------
feature_extractor = DenseNet201(weights='imagenet', include_top=False, pooling=None, input_shape=(224,224,3))

# Load trained classifier (expects 7x7x1920 input)
classifier = load_model("/content/feature_odir_densenet201.hdf5")

# Define class names
classes = ['A_category', 'D_category', 'G_category', 'H_category', 'M_category']  # update as per dataset

# -----------------------------
# Image preprocessing
# -----------------------------
def load_img_array(img_path, target_size=(224,224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Predict single image
# -----------------------------
def predict_from_image_path(img_path):
    img_array = load_img_array(img_path)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    pred_probs = classifier.predict(features)
    pred_index = np.argmax(pred_probs)
    return pred_index, classes[pred_index], features, pred_probs

# -----------------------------
# Grad-CAM visualization
# -----------------------------
def grad_cam(img_path, layer_name=None):
    """Compute and display Grad-CAM for a feature-based model."""
    # Predict
    pred_index, pred_class, features, pred_probs = predict_from_image_path(img_path)
    
    # Select last conv layer from feature extractor
    if layer_name is None:
        # pick the last conv layer
        for layer in reversed(feature_extractor.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break
    
    last_conv_layer = feature_extractor.get_layer(layer_name)
    n_channels = last_conv_layer.output.shape[-1]

    # Gradient of predicted class w.r.t. last conv layer
    with K.get_session().graph.as_default():
        class_output = K.sum(classifier.output[:, pred_index])  # sum to get scalar
        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0,1,2))
        iterate = K.function([feature_extractor.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_val, conv_layer_output_val = iterate(preprocess_input(load_img_array(img_path)))
    
    # Multiply each channel by its importance
    for i in range(n_channels):
        conv_layer_output_val[:,:,i] *= pooled_grads_val[i]
    
    # Heatmap
    heatmap = np.mean(conv_layer_output_val, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Superimpose on original image
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.5 + img

    plt.figure(figsize=(12,6))
    plt.imshow(cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {pred_class}")
    plt.axis('off')
    plt.show()

# -----------------------------
# Predict folder with optional Grad-CAM
# -----------------------------
def predict_folder(folder_path, use_gradcam=False):
    results = []
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(supported_ext):
            image_path = os.path.join(folder_path, filename)
            pred_index, pred_class, _, _ = predict_from_image_path(image_path)
            results.append((filename, pred_class))
            print(f"{filename}: {pred_class}")
            if use_gradcam:
                grad_cam(image_path)
    
    return results

# -----------------------------
# Example usage
# -----------------------------
folder_path = '/content/test_images'
results = predict_folder(folder_path, use_gradcam=True)

#########################################################
#SIR CODE
#########################################################

