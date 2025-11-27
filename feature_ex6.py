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
classes = ['A_category', 'D_category', 'G_category', 'H_category', 'M_category']  # update as per dataset

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
def predict_from_image_path(img_path):
    """Predict the class of a single image using feature extractor + classifier."""
    img_array = load_img_array(img_path)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)  # shape (1,7,7,1920)
    pred_index = np.argmax(classifier.predict(features))
    return pred_index, classes[pred_index]

# -----------------------------
# Predict all images in a folder
# -----------------------------
def predict_folder(folder_path):
    """Predict all images inside a folder."""
    results = []
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(supported_ext):
            image_path = os.path.join(folder_path, filename)
            pred_index, pred_class = predict_from_image_path(image_path)
            results.append((filename, pred_class))
            print(f"{filename}: {pred_class}")

    return results

# -----------------------------
# Example usage
# -----------------------------
folder_path = '/content/test_images'  # folder containing test images
results = predict_folder(folder_path)
