
#Ex-4. Image Classification Using Pre-trained CNN models

from tensorflow.keras.applications import VGG16,VGG19,ResNet50,InceptionV3,Xception,imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np

model_name = "ResNet50"
path = "bird.jpg"

models = {
    "VGG16":(VGG16,(224,224),imagenet_utils.preprocess_input),
    "VGG19":(VGG19,(224,224),imagenet_utils.preprocess_input),
    "ResNet50":(ResNet50,(224,224),imagenet_utils.preprocess_input),
    "InceptionV3":(InceptionV3,(299,299),preprocess_input),
    "xception":(Xception,(299,299),preprocess_input)
}

model,size,preprocess = models[model_name]
Model = model(weights = "imagenet")
img = load_img(path,target_size = size)
arr = img_to_array(img)
img = np.expand_dims(arr,0)
img = preprocess(img)

preds = Model.predict(img)
decoded = imagenet_utils.decode_predictions(preds,top= 3)[0]

for (i,(id,label,prob)) in enumerate(decoded):
    print(f"{i+1} {label} : {prob*100:.2f}%")

#==========================================================================
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
# load the model
model = VGG16()
# load an image from file
image = load_img('mug.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = np.expand_dims(image, axis=0)
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
