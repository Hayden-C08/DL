#===================================================================
#summmarize all layers
#===================================================================
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
# load the model
model = VGG16()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# summarize filter shapes
print(model.layers)
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)


#====================================================================
#filter visualization
#====================================================================
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np

# load model
model = VGG16()

# get filters of first conv layer
filters, biases = model.layers[1].get_weights()

# normalize filters for visualization
f = (filters - filters.min()) / (filters.max() - filters.min())

# number of filters
n_filters = f.shape[3]

grid_size = int(np.ceil(np.sqrt(n_filters)))

plt.figure(figsize=(12, 12))
for i in range(n_filters):
    plt.subplot(grid_size, grid_size, i+1)
    plt.imshow(f[:, :, :, i])
    plt.axis('off')


plt.show()

#=================================================================================
#feature map visualization
#=================================================================================
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# Load VGG16
model = VGG16()

# Output of first conv layer (block1_conv1)
model = Model(inputs=model.inputs, outputs=model.layers[1].output)

# Load image
img = load_img("bird.jpg", target_size=(224, 224))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# Get feature maps
feature_maps = model.predict(img)

# Number of feature channels
num_maps = feature_maps.shape[-1]
grid = int(np.ceil(np.sqrt(num_maps)))

plt.figure(figsize=(12, 12))

for i in range(num_maps):
    plt.subplot(grid, grid, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()


#=====================================================================
#feature maps from multiple blocks
#=====================================================================
print("\n=== LAYER VIS 4 — MULTIPLE CONV BLOCKS FEATURE MAPS ===\n")

ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]
multi_model = Model(inputs=model.inputs, outputs=outputs)

# Extract feature maps
feature_maps = multi_model.predict(img)

# Visualize maps from each block
square = 8
for layer_index, fmap in enumerate(feature_maps):
    ix = 1
    plt.figure(figsize=(12, 12))
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
    plt.suptitle(f"Feature Maps — Conv Block {layer_index + 1}", fontsize=14)
    plt.show()

#====================================================================================
# 5️Simple Sequential Dense Model Visualization (Manual)
# ==========================================================
print("\n=== LAYER VIS 5 — SIMPLE SEQUENTIAL MODEL (DRAWN) ===\n")

simple_model = Sequential()
simple_model.add(Dense(2, input_dim=1, activation='relu'))
simple_model.add(Dense(1, activation='sigmoid'))
simple_model.summary()

# Draw using Matplotlib
plt.figure(figsize=(8, 3))
plt.text(0.05, 0.6, 'Input Layer (1 neuron)', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))
plt.text(0.4, 0.6, 'Dense Layer (2, ReLU)', fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
plt.text(0.75, 0.6, 'Output Layer (1, Sigmoid)', fontsize=12, bbox=dict(facecolor='lightcoral', alpha=0.5))
plt.arrow(0.22, 0.62, 0.12, 0, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.arrow(0.58, 0.62, 0.12, 0, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.axis('off')
plt.title("Simple Sequential Model Visualization", fontsize=14)
plt.show()
