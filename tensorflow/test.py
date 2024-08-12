import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import shutil
from collections import Counter
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,\
    Dense, Layer, Reshape, InputLayer, Flatten, Input, MaxPooling2D
from alibi_detect.od import OutlierAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image

from sklearn.metrics import accuracy_score


# Prepare image data 
def img_to_np(path, resize = True, extract_labels=False):  
    img_array = []
    labels = []
    fpaths = glob.glob(path, recursive=True)
    for fname in fpaths:
        if(extract_labels): 
            if '_bad' in os.path.basename(fname):
                labels.append(1)  # 1 for outlier
            else:
                labels.append(0)  # 0 for non-outlier
        img = Image.open(fname).convert("L")
        if(resize): img = img.resize((64,64))
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    if(extract_labels): return images, np.array(labels)
    return images

path_train = r'C:\Users\Ossi\Desktop\ImageML\elpv-dataset\train_without_bad_images\**\*'
path_test = r'C:\Users\Ossi\Desktop\ImageML\elpv-dataset\test_images\**\*'

train = img_to_np(path_train)
test, test_labels = img_to_np(path_test, extract_labels=True)
train = train.astype('float32') / 255.0
test = test.astype('float32') / 255.0

# Prepare image data 
def img_to_np(path, resize = True, extract_labels=False):  
    img_array = []
    labels = []
    fpaths = glob.glob(path, recursive=True)
    for fname in fpaths:
        if(extract_labels): 
            if '_bad' in os.path.basename(fname):
                labels.append(1)  # 1 for outlier
            else:
                labels.append(0)  # 0 for non-outlier
        img = Image.open(fname).convert("L") # Grayscale when using "RGB" you have to change the encoder and decoder 
        if(resize): img = img.resize((64,64))
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    if(extract_labels): return images, np.array(labels)
    return images

path_train = r'C:\Users\Ossi\Desktop\ImageML\elpv-dataset\train_without_bad_images\**\*'
path_test = r'C:\Users\Ossi\Desktop\ImageML\elpv-dataset\test_images\**\*'

train = img_to_np(path_train)
test, test_labels = img_to_np(path_test, extract_labels=True)
train = train.astype('float32') / 255.0
test = test.astype('float32') / 255.0
# Reshape to include the channel dimension
train = np.expand_dims(train, axis=-1)
test = np.expand_dims(test, axis=-1)

# Model parameters
encoding_dim = 1024
dense_dim = [8, 8, 128]

# Define the encoder
encoder_net = tf.keras.Sequential([
    Input(shape=(64, 64, 1)),  # Updated input shape for grayscale images
    Conv2D(64, 4, strides=2, padding='same', activation='relu'),
    Conv2D(128, 4, strides=2, padding='same', activation='relu'),
    Conv2D(512, 4, strides=2, padding='same', activation='relu'),
    Flatten(),
    Dense(encoding_dim)
])

# Define the decoder
decoder_net = tf.keras.Sequential([
    Input(shape=(encoding_dim,)),
    Dense(np.prod(dense_dim)),
    Reshape(target_shape=dense_dim),
    Conv2DTranspose(256, 4, strides=2, padding='same', activation='relu'),
    Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
    Conv2DTranspose(1, 4, strides=2, padding='same', activation='sigmoid')  # Updated output channels to 1
])

# Create the OutlierAE model
od = OutlierAE(
    threshold=0.001,
    encoder_net=encoder_net,
    decoder_net=decoder_net
)

# Compile and train the model
adam = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Corrected keyword
od.fit(train, epochs=10, verbose=True, optimizer=adam)