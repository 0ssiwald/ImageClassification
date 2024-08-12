import os
import random

# Set up the directory paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
image_good_folder = os.path.join(parent_dir, 'non_defective_images')
image_bad_folder = os.path.join(parent_dir, 'defective_images')

# Get a list of image files in each directory
image_files_good = [f for f in os.listdir(image_good_folder) if os.path.isfile(os.path.join(image_good_folder, f))]
image_files_bad = [f for f in os.listdir(image_bad_folder) if os.path.isfile(os.path.join(image_bad_folder, f))]

# Print the initial number of images
print(f"Number of non-defective images: {len(image_files_good)}")
print(f"Number of defective images: {len(image_files_bad)}")

# Determine the folder with more images and the difference
if len(image_files_good) > len(image_files_bad):
    excess_images = image_files_good
    excess_folder = image_good_folder
    target_count = len(image_files_bad)
else:
    excess_images = image_files_bad
    excess_folder = image_bad_folder
    target_count = len(image_files_good)

# Calculate how many images to remove
images_to_remove = len(excess_images) - target_count

# Randomly select images to remove
images_to_delete = random.sample(excess_images, images_to_remove)

# Delete the selected images
for image in images_to_delete:
    os.remove(os.path.join(excess_folder, image))
    print(f"Deleted {image} from {excess_folder}")

# Print the final number of images
final_count_good = len([f for f in os.listdir(image_good_folder) if os.path.isfile(os.path.join(image_good_folder, f))])
final_count_bad = len([f for f in os.listdir(image_bad_folder) if os.path.isfile(os.path.join(image_bad_folder, f))])

print(f"Final number of non-defective images: {final_count_good}")
print(f"Final number of defective images: {final_count_bad}")