# https://towardsdatascience.com/simplifing-image-outlier-detection-with-alibi-detect-6aea686bf7ba
# source venv/bin/activate

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, Input
from alibi_detect.od import OutlierAE
from alibi_detect.utils.visualize import plot_feature_outlier_image
from train_test_split import load_and_split_images

# Set parent directory and image folder paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
image_folder = os.path.join(parent_dir, 'non_defective_images')

# Load and split images
train, test = load_and_split_images(image_folder, train_ratio=0.8)
train = train.astype('float32') / 255.0
test = test.astype('float32') / 255.0

# Model parameters
encoding_dim = 1024
dense_dim = [8, 8, 128]

# Define the encoder
encoder_net = tf.keras.Sequential([
    Input(shape=train[0].shape),
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
    Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
])

# Create the OutlierAE model
od = OutlierAE(
    threshold=0.001,
    encoder_net=encoder_net,
    decoder_net=decoder_net
)

# Compile and train the model
adam = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Corrected keyword
od.fit(train, epochs=100, verbose=True, optimizer=adam)

# Infer threshold on test set
od.infer_threshold(test, threshold_perc=95)

# Test the model on the test set
preds = od.predict(test, outlier_type='instance',
                   return_instance_score=True,
                   return_feature_score=True)

# Get reconstructions
recon = od.ae(test).numpy()

# Visualize outliers
plot_feature_outlier_image(preds, test, 
                           X_recon=recon,  
                           max_instances=5,
                           outliers_only=True,
                           figsize=(15, 15))





'''
for i, fpath in enumerate(glob.glob(path_test)):
    if(preds['data']['is_outlier'][i] == 1):
        source = fpath
        shutil.copy(source, 'img\\') 
        
filenames = [os.path.basename(x) for x in glob.glob(path_test, recursive=True)]

dict1 = {'Filename': filenames,
     'instance_score': preds['data']['instance_score'],
     'is_outlier': preds['data']['is_outlier']}
     
df = pd.DataFrame(dict1)
df_outliers = df[df['is_outlier'] == 1]

print(df_outliers)
'''