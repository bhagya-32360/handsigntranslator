import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer

# Set paths to your dataset (adjust if needed)
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

print(f"Classes found in training data: {train_data.class_indices}")
print(f"Number of training images: {train_data.samples}")

# Define the CNN model using Input layer as recommended
model = Sequential([
    tf.keras.Input(shape=(*image_size, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(29, activation='softmax')  # 26 classes: A-Z and del,nothing,space
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
model.fit(train_data, epochs=10)  # Adjust epochs as needed

# Save the trained model in new recommended format
model.save('asl_model.keras')
print("Model training complete and saved as asl_model.keras")

# ------- Custom test data loading -------

test_images = []
test_labels = []

print(f"Loading test images from: {os.path.abspath(test_dir)}")
try:
    files = os.listdir(test_dir)
    print(f"Found {len(files)} files in test directory.")
except Exception as e:
    print(f"Error accessing test directory: {e}")
    files = []

for filename in files:
    # Case-insensitive check for image extensions
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        label = filename[0].upper()  # Assuming filename starts with letter label
        img_path = os.path.join(test_dir, filename)
        try:
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0
            test_images.append(img_array)
            test_labels.append(label)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")

test_images = np.array(test_images)
print(f"Loaded {len(test_images)} test images.")

if len(test_images) == 0:
    print("Warning: No test images loaded! Check your test directory and filenames.")
else:
    # One-hot encode test labels
    lb = LabelBinarizer()
    test_labels_encoded = lb.fit_transform(test_labels)
    
    print(f"Test labels classes: {lb.classes_}")
    print(f"Shape of test images array: {test_images.shape}")
    print(f"Shape of test labels array: {test_labels_encoded.shape}")
    
    # Evaluate the model on test data
    loss, accuracy = model.evaluate(test_images, test_labels_encoded)
    print(f"Test accuracy: {accuracy*100:.2f}%")
