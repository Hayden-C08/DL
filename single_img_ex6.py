import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing import image

# Load feature extractor (DenseNet201 without top)
feature_extractor = DenseNet201(weights='imagenet', include_top=False, pooling=None, input_shape=(224,224,3))

# Load your trained classifier (expects 7x7x1920 input)
classifier = load_model("/content/feature_odir_densenet201.hdf5")

# Define your class names
classes = ['A_category', 'D_category', 'G_category', 'H_category', 'M_category']  # update as per dataset

def load_img(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_from_image_path(img_path):
    img_array = load_img(img_path)
    img_array = preprocess_input(img_array)

    # Extract features
    features = feature_extractor.predict(img_array)  # shape (1,7,7,1920)

    # Classify
    pred = np.argmax(classifier.predict(features))
    return pred, classes[pred]

# Example
print(predict_from_image_path('/content/2040_left.jpg'))