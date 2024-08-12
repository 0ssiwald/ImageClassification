import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import glob


def load_and_split_images(image_folder, train_ratio=0.8, resize=True, convert=True):
    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # load an image as a numpy array
    def load_image(filename):
        with Image.open(os.path.join(image_folder, filename)) as img:
            if(convert): img = img.convert("RGB")
            if(resize): img = img.resize((64,64))
            return np.array(img)
    
    # Load all images
    images = [load_image(f) for f in image_files]
    
    # Convert list of images to a numpy array
    images = np.array(images)
    
    # Split the data into train and test sets
    train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)
    
    return train_images, test_images


# Usage example:
if __name__ == '__main__':
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    image_folder = os.path.join(parent_dir, 'non_defective_images')

    train_images, test_images = load_and_split_images(image_folder, train_ratio=0.8)
    
    print(f"Number of training images: {len(train_images)}")
    print(f"Number of test images: {len(test_images)}")
    print(f"Shape of each image: {train_images[0].shape}")

    # For saving in train_test_folder