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
    pred_index, pred_class, features, pred_probs = predict_from_image_path(img_path)

    # pick last conv layer
    if layer_name is None:
        for layer in reversed(feature_extractor.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break

    last_conv_layer = feature_extractor.get_layer(layer_name)
    n_channels = last_conv_layer.output.shape[-1]

    with K.get_session().graph.as_default():
        class_output = K.sum(classifier.output[:, pred_index])
        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0,1,2))
        iterate = K.function([feature_extractor.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_val, conv_layer_output_val = iterate(preprocess_input(load_img_array(img_path)))

    for i in range(n_channels):
        conv_layer_output_val[:,:,i] *= pooled_grads_val[i]

    heatmap = np.mean(conv_layer_output_val, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

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
# Predict folder with optional Grad-CAM (NOW SUPPORTS SUBFOLDERS)
# -----------------------------
def predict_folder(folder_path, use_gradcam=False):
    results = []
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    # 🔥 Walk through all subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in sorted(files):
            if filename.lower().endswith(supported_ext):

                image_path = os.path.join(root, filename)

                pred_index, pred_class, _, _ = predict_from_image_path(image_path)
                results.append((image_path, pred_class))

                print(f"{image_path}: {pred_class}")

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

import os
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras import backend as K
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

K.set_learning_phase(0)

# ---------------------------------------
# 1. LOAD MODELS (feature extractor + head)
# ---------------------------------------
feature_extractor = load_model("feature_extractor.h5")
classifier = load_model("classifier.h5")

classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# ---------------------------------------
# 2. PREPROCESS FUNCTION
# ---------------------------------------
def preprocess_input(img):
    x = img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)

# ---------------------------------------
# 3. FEATURE EXTRACTION + CLASSIFICATION
# ---------------------------------------
def predict_image(img):
    x = preprocess_input(img)
    feat = feature_extractor.predict(x)     # (1, 7, 7, 1920)
    pred = classifier.predict(feat)[0]
    cls = np.argmax(pred)
    return cls, classes[cls], pred

def predict_from_image_path(path):
    img = load_img(path, target_size=(224, 224))
    return predict_image(img)

# ---------------------------------------
# 4. GRAD-CAM FOR FEATURE EXTRACTION MODEL
# ---------------------------------------
def grad_CAM(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = preprocess_input(img)

    # Forward pass to get feature map
    conv_layer = feature_extractor.get_layer("conv5_block32_concat")
    heat_model = keras.models.Model(
        inputs=feature_extractor.input,
        outputs=[conv_layer.output, feature_extractor.output]
    )

    conv_output, preds = heat_model.predict(x)

    # Predicted class
    pred_class = np.argmax(classifier.predict(preds))

    # Gradient wrt feature map
    class_output = classifier.output[:, pred_class]
    grads = K.gradients(class_output, conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # Get numpy values
    iterate = K.function(
        [feature_extractor.input],
        [pooled_grads, conv_layer.output[0]]
    )
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # Weight feature maps by gradients
    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # Create heatmap
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Read original image
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (224, 224))

    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap
    superimposed_img = heatmap * 0.4 + img_cv

    # Display
    plt.figure(figsize=(10, 6))
    plt.imshow(superimposed_img[..., ::-1])
    plt.axis("off")
    plt.show()

# ---------------------------------------
# 6. TEST PREDICTION + GRAD-CAM
# ---------------------------------------
path = "test_image.jpeg"
pred, name, prob = predict_from_image_path(path)
print("Prediction:", pred, name)
grad_CAM(path)

# ---------------------------------------
# 7. CHECK MULTIPLE IMAGES
# ---------------------------------------
for i, c in enumerate(classes):
    folder = './simple/test/' + c + '/'
    for file in os.listdir(folder):
        if file.endswith(".jpeg"):
            image_path = folder + file
            p, cname, _ = predict_from_image_path(image_path)

            if p == i:
                print(file, p, cname)
            else:
                print(file, p, cname, "** INCORRECT **")
                grad_CAM(image_path)


