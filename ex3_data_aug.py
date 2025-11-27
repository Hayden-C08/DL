# ============================================================
# 1)HORIZONTAL AUG
# ============================================================
from numpy import expand_dims
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from matplotlib import pyplot as plt

img = load_img('bird.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)

datagen = ImageDataGenerator(width_shift_range=[-200, 200])
it = datagen.flow(samples, batch_size=1)

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = it.next()
    plt.imshow(batch[0].astype('uint8'))
    plt.axis('off')
plt.show()


# ============================================================
# 2) VERTICAL AUG
# ============================================================
img = load_img('bird.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)

datagen = ImageDataGenerator(height_shift_range=0.5)
it = datagen.flow(samples, batch_size=1)

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = it.next()
    plt.imshow(batch[0].astype('uint8'))
    plt.axis('off')
plt.show()


# ============================================================
# 3) HORIZONTAL FLIP AUGMENTATION
# ============================================================
img = load_img('bird.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)

datagen = ImageDataGenerator(horizontal_flip=True)
it = datagen.flow(samples, batch_size=1)

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = it.next()
    plt.imshow(batch[0].astype('uint8'))
    plt.axis('off')
plt.show()


# ============================================================
# 4) ROTATION AUGMENTATION
# ============================================================
img = load_img('bird.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)

datagen = ImageDataGenerator(rotation_range=90)
it = datagen.flow(samples, batch_size=1)

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = it.next()
    plt.imshow(batch[0].astype('uint8'))
    plt.axis('off')
plt.show()


# ============================================================
# 5) BRIGHTNESS AUGMENTATION
# ============================================================
img = load_img('bird.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)

datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])
it = datagen.flow(samples, batch_size=1)

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = it.next()
    plt.imshow(batch[0].astype('uint8'))
    plt.axis('off')
plt.show()


# ============================================================
# 6) ZOOM AUGMENTATION
# ============================================================
img = load_img('bird.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)

datagen = ImageDataGenerator(zoom_range=0.5)
it = datagen.flow(samples, batch_size=1)

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = it.next()
    plt.imshow(batch[0].astype('uint8'))
    plt.axis('off')
plt.show()
