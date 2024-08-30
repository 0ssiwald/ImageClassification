import os
import shutil
import random

# Set up the directory paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
image_good_folder = os.path.join(parent_dir, 'non_defective_images')
image_bad_folder = os.path.join(parent_dir, 'defective_images')
train_wb_folder = os.path.join(parent_dir, 'train_with_bad_images')
train_nb_folder = os.path.join(parent_dir, 'train_without_bad_images')
test_folder = os.path.join(parent_dir, 'test_images')
train_ng_folder = os.path.join(parent_dir, 'train_without_good_images')

# Create the target directories and subdirectories if they don't exist
os.makedirs(os.path.join(train_wb_folder, 'good'), exist_ok=True)
os.makedirs(os.path.join(train_wb_folder, 'bad'), exist_ok=True)
os.makedirs(os.path.join(train_nb_folder, 'good'), exist_ok=True)
os.makedirs(os.path.join(train_ng_folder, 'bad'), exist_ok=True)
os.makedirs(os.path.join(test_folder, 'good'), exist_ok=True)
os.makedirs(os.path.join(test_folder, 'bad'), exist_ok=True)

# Get a list of image files in the source directories
image_files_good = [f for f in os.listdir(image_good_folder) if os.path.isfile(os.path.join(image_good_folder, f))]
image_files_bad = [f for f in os.listdir(image_bad_folder) if os.path.isfile(os.path.join(image_bad_folder, f))]

# Shuffle the image files to ensure random selection
random.shuffle(image_files_good)
random.shuffle(image_files_bad)

# Calculate the split index
split_index_good = int(len(image_files_good) * 0.85)
split_index_bad = int(len(image_files_bad) * 0.85)

# Split the images into training and testing sets
train_good_images = image_files_good[:split_index_good]
test_good_images = image_files_good[split_index_good:]
train_bad_images = image_files_bad[:split_index_bad]
test_bad_images = image_files_bad[split_index_bad:]


def copy_images(image_list, destination_folder, source_folder):
    for image in image_list:
        # Original file path
        original_path = os.path.join(source_folder, image)

        # Destination file path
        new_path = os.path.join(destination_folder, image)

        # Copy the file to the new destination
        shutil.copy(original_path, new_path)

# Copy the images into the respective folders
copy_images(train_good_images, os.path.join(train_wb_folder, 'good'), image_good_folder)
copy_images(train_good_images, os.path.join(train_nb_folder, 'good'), image_good_folder)
copy_images(train_bad_images, os.path.join(train_wb_folder, 'bad'), image_bad_folder)
copy_images(train_bad_images, os.path.join(train_ng_folder, 'bad'), image_bad_folder)
copy_images(test_good_images, os.path.join(test_folder, 'good'), image_good_folder)
copy_images(test_bad_images, os.path.join(test_folder, 'bad'), image_bad_folder)