#########################################################
#MY CODE
###############################################################


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from pickle import load

# ------------------------------
# Load trained captioning model and tokenizer
# ------------------------------
model = load_model("/content/caption_odir_densnet201.hdf5")
model.summary()

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
# Predict caption for a new image
# ------------------------------
image_path = "/content/2040_left.jpg"  # Replace with your image path
photo_feature = extract_features(image_path)
caption = generate_caption(model, tokenizer, photo_feature, max_length)

print("Generated Caption:", caption)

########################################################################
#SIR CODE
########################################################################
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import cv2 
# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('model-ep004-loss3.572-val_loss3.833.h5')
# load and prepare the photograph
photo = extract_features('example.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
image = load_img('example.jpg')
image = img_to_array(image)
orig = cv2.imread('example.jpg')
cv2.putText(orig,description,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),2,cv2.LINE_AA)
cv2.imshow("CAPTION GENERATION", orig)

cv2.waitKey(0)
print(description)

# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen 
