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
# Simple prediction pipeline
# -----------------------------
def preprocess_image(image_path):
    """Load and preprocess image for DenseNet201."""
    img = load_img(image_path, target_size=(224, 224))
    print(img)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize
    return img_array

def predict_image(image_path):
    """Predict the class of a single image."""
    img_array = preprocess_image(image_path)
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = classes[pred_index]
    confidence = preds[0][pred_index]
    return pred_class, confidence

# -----------------------------
# Example usage
# -----------------------------
image_path = '/content/2040_left.jpg'  # replace with your image path
pred_class, confidence = predict_image(image_path)
print(f"Predicted Class: {pred_class} ({confidence*100:.2f}% confidence)")