import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing import image

# -----------------------------
# Load feature extractor (DenseNet201 without top)
# -----------------------------
feature_extractor = DenseNet201(weights='imagenet', include_top=False, pooling=None, input_shape=(224,224,3))

# Load trained classifier (expects 7x7x1920 input)
classifier = load_model("/content/feature_odir_densenet201.hdf5")

# Define class names
classes = ['A_category', 'D_category', 'G_category', 'H_category', 'M_category']


# -----------------------------
# Image preprocessing
# -----------------------------
def load_img_array(img_path, target_size=(224,224)):
    """Load image and convert to array."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -----------------------------
# Predict single image
# -----------------------------
def predict_single_image(img_path):
    """Predict one image using feature extractor + classifier."""
    img_array = load_img_array(img_path)
    img_array = preprocess_input(img_array)

    # extract features (1, 7, 7, 1920)
    features = feature_extractor.predict(img_array)

    # classifier predictions (probabilities)
    preds = classifier.predict(features)

    pred_index = np.argmax(preds)
    pred_class = classes[pred_index]
    confidence = float(preds[0][pred_index])

    return pred_class, confidence


# -----------------------------
# Predict all images in folder (supports subfolders)
# -----------------------------
def predict_folder(folder_path):
    """Predict all images inside subfolders recursively."""
    results = []
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for root, dirs, files in os.walk(folder_path):
        for filename in sorted(files):
            if filename.lower().endswith(supported_ext):

                image_path = os.path.join(root, filename)
                pred_class, confidence = predict_single_image(image_path)

                results.append((image_path, pred_class, confidence))
                print(f"{image_path}: {pred_class} ({confidence * 100:.2f}%)")

    return results


# -----------------------------
# Example usage
# -----------------------------
folder_path = "/content/test_images"
results = predict_folder(folder_path)

