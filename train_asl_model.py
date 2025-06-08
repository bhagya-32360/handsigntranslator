import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Set paths to your dataset
train_dir = 'asl_alphabet_train'
test_dir = '../asl_alphabet_test'

image_size = (64, 64)

# Image preprocessing and augmentation for training
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

# Print classes and number of images in training data
print("Classes found in training data:", train_data.class_indices)
print(f"Number of training images: {train_data.samples}")

# Define the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(*image_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(29, activation='softmax')  # 26 classes: A-Z
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
model.fit(train_data, epochs=10)

# Save the trained model
model.save('asl_model.h5')
print(" Model training complete and saved as asl_model.h5")

# ------- Custom test data loading -------

test_images = []
test_labels = []

for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        label = filename[0].upper()  # Assuming filename starts with letter label
        img_path = os.path.join(test_dir, filename)
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0
        test_images.append(img_array)
        test_labels.append(label)

test_images = np.array(test_images)
print(f"Loaded {len(test_images)} test images.")

# Map test labels to class indices based on training data
class_indices = train_data.class_indices
num_classes = len(class_indices)

try:
    test_label_indices = [class_indices[label] for label in test_labels]
except KeyError as e:
    print(f"Label {e} not found in training classes. Check test label names.")
    raise

# Convert to one-hot encoding
test_labels_onehot = tf.keras.utils.to_categorical(test_label_indices, num_classes=num_classes)

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_images, test_labels_onehot)
print(f"Test accuracy: {accuracy*100:.2f}%")


