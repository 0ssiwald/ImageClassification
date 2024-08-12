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

# Create the target directories if they don't exist
os.makedirs(train_wb_folder, exist_ok=True)
os.makedirs(train_nb_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(train_ng_folder, exist_ok=True)

# Get a list of image files in the source directory
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


def copy_and_rename_images(image_list, destination_folder, source_folder, name_ending):
    for image in image_list:
        # Original file path
        original_path = os.path.join(source_folder, image)

        # Split the file name and extension
        name, ext = os.path.splitext(image)

        # Create new file name with "_working" appended
        new_name = f"{name}_{name_ending}{ext}"

        # New file path
        new_path = os.path.join(destination_folder, new_name)

        # Copy and rename the file
        shutil.copy(original_path, new_path)

# Copy and rename the images
copy_and_rename_images(train_good_images, train_wb_folder, image_good_folder, 'good')
copy_and_rename_images(train_good_images, train_nb_folder, image_good_folder, 'good')
copy_and_rename_images(train_bad_images, train_wb_folder, image_bad_folder, 'bad')
copy_and_rename_images(train_bad_images, train_ng_folder, image_bad_folder, 'bad')
copy_and_rename_images(test_good_images, test_folder, image_good_folder, 'good')
copy_and_rename_images(test_bad_images, test_folder, image_bad_folder, 'bad')

