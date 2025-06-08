import os
import shutil
import random

# Path to the original dataset folder (you already have this)
source_dir = "asl_alphabet_train"
# New folder to store 100 images per class
target_dir = "asl_alphabet_100"

# Create the target folder if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Loop over each class (A, B, C, ..., space, nothing, del)
for label in os.listdir(source_dir):
    label_path = os.path.join(source_dir, label)
    if not os.path.isdir(label_path):
        continue

    target_label_path = os.path.join(target_dir, label)
    os.makedirs(target_label_path, exist_ok=True)

    # Pick 100 random images from the class
    all_images = os.listdir(label_path)
    selected_images = random.sample(all_images, 100)

    for image in selected_images:
        src_image = os.path.join(label_path, image)
        dst_image = os.path.join(target_label_path, image)
        shutil.copy(src_image, dst_image)

print("100 images per class copied successfully to 'asl_alphabet_100'")
