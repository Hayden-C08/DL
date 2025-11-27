import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

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
# Grad-CAM
# -----------------------------
def grad_cam(image_path):
    """Generate Grad-CAM heatmap for a single image."""
    img = preprocess_image(image_path)
    preds = model.predict(img)
    pred_index = np.argmax(preds)
    
    # Last convolutional layer in DenseNet201
    last_conv_layer = model.get_layer('conv5_block32_concat')
    grads = K.gradients(model.output[:, pred_index], last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0,1,2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])
    
    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Load original image
    original_img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)
    
    plt.figure(figsize=(12,6))
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# -----------------------------
# Predict all images in a folder
# -----------------------------
def predict_folder(folder_path, visualize_gradcam=False):
    """Predict all images inside a folder."""
    results = []
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(supported_ext):
            image_path = os.path.join(folder_path, filename)
            pred_class, confidence = predict_image(image_path)
            results.append((filename, pred_class, confidence))
            print(f"{filename}: {pred_class} ({confidence*100:.2f}%)")
            if visualize_gradcam:
                grad_cam(image_path)

    return results

# -----------------------------
# Example usage
# -----------------------------
folder_path = '/content/test_images'  # folder containing images
results = predict_folder(folder_path, visualize_gradcam=True)
