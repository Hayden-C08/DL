import os
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Load DenseNet201 as feature extractor
feature_extractor = DenseNet201(weights='imagenet', include_top=False, pooling=None, input_shape=(224,224,3))

# 2️⃣ Load your trained classifier
classifier = load_model("/content/densenet201_classifier_full.hdf5")

# 3️⃣ Define classes
classes = ['A_category', 'D_category', 'G_category', 'H_category', 'M_category']

# 4️⃣ Load & preprocess image
def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# 5️⃣ Extract features for a folder of test images
def extract_features_from_folder(test_dir):
    X_test, y_test = [], []
    for label in os.listdir(test_dir):
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(label_dir, fname)
            img_array = load_and_preprocess(img_path)
            features = feature_extractor.predict(img_array)
            X_test.append(features[0])
            y_test.append(classes.index(label))
    return np.array(X_test), np.array(y_test)

# 6️⃣ Run feature extraction and evaluation
test_dir = "/content/test_images"  # folder structured like training folder
X_test, y_test = extract_features_from_folder(test_dir)

# Predict
y_pred = classifier.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Accuracy
accuracy = np.sum(y_pred_classes == y_test) / len(y_test)
print(f"✅ Test Accuracy: {accuracy*100:.2f}%")

# Classification report & confusion matrix
print(classification_report(y_test, y_pred_classes, target_names=classes))
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()