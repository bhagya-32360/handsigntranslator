import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Paths
model_path = 'asl_model.keras'
test_dir = 'asl_alphabet_train'  # Your test folder containing class subfolders

# Load the trained model
model = load_model(model_path)

print("Model loaded successfully.")
print("Loading test images ...")

# Prepare test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict on test data
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# Get class labels
class_labels = list(test_data.class_indices.keys())

# Print classification results
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
