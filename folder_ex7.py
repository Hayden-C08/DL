import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import load

# ------------------------------
# Load trained captioning model and tokenizer
# ------------------------------
model = load_model("/content/caption_odir_densnet201.hdf5")

with open('/content/tokenizer_odir_densenet201.pkl', 'rb') as handle:
    tokenizer = load(handle)

# ------------------------------
# Load DenseNet201 for feature extraction
# ------------------------------
base_model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path):
    """Extract DenseNet201 features for a given image."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = base_model.predict(image, verbose=0)
    return feature

# ------------------------------
# Caption generation helpers
# ------------------------------
max_length = 13  # same as training

def word_for_id(integer, tokenizer):
    """Map an integer to a word using the tokenizer."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, photo, max_length):
    """Generate a caption for an image feature."""
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break

    caption = in_text.replace('startseq ', '').replace(' endseq', '')
    return caption

# ------------------------------
# Predict captions for all images in a folder
# ------------------------------
def caption_folder(folder_path):
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(supported_ext):
            img_path = os.path.join(folder_path, filename)

            # Extract features + generate caption
            features = extract_features(img_path)
            caption = generate_caption(model, tokenizer, features, max_length)

            print(f"{filename} --> {caption}")

# ------------------------------
# Example: Test folder path
# ------------------------------
test_folder = "/content/test_images"   # update to your actual folder
caption_folder(test_folder)
