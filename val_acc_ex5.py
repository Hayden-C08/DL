import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# 1. Load the saved model
# -----------------------------
model_path = '/content/best_model_densenet201.h5'  # update if needed
model = load_model(model_path)
model.summary()

# -----------------------------
# 2. Set up test data generator
# -----------------------------
test_dir = '/kaggle/input/odir5k/odr/Testing Images/'  # same format as training

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False   # important: don’t shuffle when evaluating
)

# -----------------------------
# 3. Evaluate model on test data
# -----------------------------
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"🧮 Test Loss: {loss:.4f}")

# -----------------------------
# 4. (Optional) Get per-class accuracy
# -----------------------------
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Get true labels and predictions
y_true = test_generator.classes
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
target_names = list(test_generator.class_indices.keys())
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=target_names))

# Confusion matrix (optional)
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()