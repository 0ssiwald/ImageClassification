import os
import shutil
import numpy as np
from elpv_reader import load_dataset

def copy_subset(destination_folder, lower_threshold=0, upper_threshold=1, source_csv=None):
    # Get the absolute path to the parent directory
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

    if source_csv is None:
        # If no source_csv is provided, use the default path
        source_csv = os.path.join(parent_dir, 'labels.csv')
    
    # Load the dataset
    images, probs, types = load_dataset(source_csv)
    
    # Create the full path for the destination folder
    destination_location = os.path.join(parent_dir, destination_folder)
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_location):
        os.makedirs(destination_location)
    
    # Get the directory of the source CSV file
    source_dir = os.path.dirname(source_csv)
    
    # Read the CSV file
    data = np.genfromtxt(source_csv, dtype=['|S19', '<f8', '|S4'], names=['path', 'probability', 'type'])
    
    # Iterate through the probabilities and corresponding image names
    for prob, image_name in zip(probs, np.char.decode(data['path'])):
        if prob >= lower_threshold and prob <= upper_threshold:
            # Construct the full path for the source image
            source_path = os.path.join(source_dir, image_name)
            
            # Construct the full path for the destination image
            destination_path = os.path.join(destination_location, os.path.basename(image_name))
            
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Copy the image
            shutil.copy2(source_path, destination_path)
    
    print(f"Finished copying images to {destination_location}")

# Usage example:
if __name__ == '__main__':
    copy_subset('defective_images', 0.2, 1.0)