import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------
# Load trained model and class labels
# -----------------------------
model_path = '/content/best_model_densenet201.hdf5'
model = load_model(model_path)

classes = ['A_category', 'D_category', 'G_category', 'H_category', 'M_category']  # update if needed

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image_path):
    """Load and preprocess image for DenseNet201."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize
    return img_array

# -----------------------------
# Predict single image
# -----------------------------
def predict_image(image_path):
    """Predict the class of a single image."""
    img_array = preprocess_image(image_path)
    preds = model.predict(img_array, verbose=0)
    pred_index = np.argmax(preds)
    pred_class = classes[pred_index]
    confidence = float(preds[0][pred_index])
    return pred_class, confidence

# -----------------------------
# Predict all images in a folder
# -----------------------------
def predict_folder(folder_path):
    """Predict all images inside subfolders too."""
    results = []
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for root, dirs, files in os.walk(folder_path):
        for filename in sorted(files):
            if filename.lower().endswith(supported_ext):
                image_path = os.path.join(root, filename)
                pred_class, confidence = predict_image(image_path)
                results.append((image_path, pred_class, confidence))
                print(f"{image_path}: {pred_class} ({confidence * 100:.2f}%)")

    return results


# -----------------------------
# Example usage
# -----------------------------
folder_path = '/content/test_images'  # folder containing test images
results = predict_folder(folder_path)
