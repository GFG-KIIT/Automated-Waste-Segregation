import tensorflow as tf
import numpy as np
import cv2
import os
import numpy as np
import pandas as pd
import imageio

import random
import matplotlib.pyplot as plt
import keras
import tensorflow.keras as K
import tensorflow.keras.backend as Kback

from keras import layers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Conv2DTranspose
from keras.layers import concatenate
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras.applications.resnet import ResNet50
from keras.applications import vgg16
from keras.applications import inception_v3
from keras.src.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications import DenseNet121, NASNetMobile, EfficientNetB0, Xception
from keras.layers import GlobalAveragePooling2D
from keras.applications.densenet import DenseNet121

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

#for importing the zip file
import zipfile

# Specifying the path to the zip file
zip_file_path = 'waste_classfication_data.zip'

from zipfile import ZipFile 

# Replace the path to your zip file and the directory where you want to extract the contents
zip_file_path = r"C:\Users\KIIT\Downloads\waste segregation\garbage_classification.zip"
extracted_dir = r"C:\Users\KIIT\Downloads\waste segregation\extracted_data" # Specify the directory where you want to extract

# Create a zip object and extract the contents to the specified directory
with ZipFile(zip_file_path, 'r') as zObject:
    zObject.extractall(path=extracted_dir)

import os

# Define the extracted directory (update this path as needed)
extracted_dir = "C:/Users/KIIT/Downloads/waste segregation/extracted_data/garbage_classification"  # Update this to your actual path

# Define class names (folder names)
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Initialize dictionary to store class sizes
class_sizes = {}

# Iterate over each class and count the number of files
for class_name in class_names:
    class_path = os.path.join(extracted_dir, class_name)
    if os.path.isdir(class_path):
        num_files = len(os.listdir(class_path))
        class_sizes[class_name] = num_files
    else:
        class_sizes[class_name] = 0
        print(f"Directory not found: {class_path}")  # Debug print

# Print the size of each class
for class_name, size in class_sizes.items():
    print(f"Class {class_name}: {size} samples")


import os
from PIL import Image

# Define the extracted directory (update this path as needed)
extracted_dir = "C:/Users/KIIT/Downloads/waste segregation/extracted_data/garbage_classification"  # Update this to your actual path

# Define class names (folder names)
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Initialize dictionary to store class shapes
class_shapes = {}

# Iterate over each class and calculate the shape of images
for class_name in class_names:
    class_path = os.path.join(extracted_dir, class_name)
    if os.path.isdir(class_path):
        shapes = []
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if os.path.isfile(file_path):
                try:
                    with Image.open(file_path) as img:
                        shapes.append(img.size)  # img.size returns (width, height)
                except Exception as e:
                    print(f"Could not open {file_path}: {e}")
        class_shapes[class_name] = shapes
    else:
        class_shapes[class_name] = []
        print(f"Directory not found: {class_path}")  # Debug print

# Print the shape of images in each class
for class_name, shapes in class_shapes.items():
    print(f"Class {class_name}: {len(shapes)} images with shapes {shapes[:5]} ...")  # Print first 5 shapes for brevity


import os
import cv2

# Define the extracted directory (update this path as needed)
extracted_dir = "C:/Users/KIIT/Downloads/waste segregation/extracted_data/garbage_classification"
resized_dir = "C:/Users/KIIT/Downloads/waste segregation/resized_dir"  # New directory for resized images

# Create the new directory structure
if not os.path.exists(resized_dir):
    os.makedirs(resized_dir)

# Define class names (folder names)
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Define the target size
target_size = (128, 128)

# Resize images and save them to the new directory
for class_name in class_names:
    
    class_path = os.path.join(extracted_dir, class_name)
    resized_class_path = os.path.join(resized_dir, class_name)
    
    if not os.path.exists(resized_class_path):
        os.makedirs(resized_class_path)

    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
                file_path = os.path.join(class_path, filename)
                img = cv2.imread(file_path)
                if img is not None:
                    resized_img = cv2.resize(img, target_size)
                    output_path = os.path.join(resized_class_path, filename)
                    cv2.imwrite(output_path, resized_img)
                else:
                    print(f"Failed to load image: {file_path}")
    else:
        print(f"Directory not found: {class_path}")

print("Image resizing completed.")



import os
from PIL import Image

resized_dir = "C:/Users/KIIT/Downloads/waste segregation/resized_dir"

# Define class names (folder names)
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Initialize dictionary to store class shapes
class_shapes = {}

# Iterate over each class and calculate the shape of images
for class_name in class_names:
    class_path = os.path.join(resized_dir, class_name)
    if os.path.isdir(class_path):
        shapes = []
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if os.path.isfile(file_path):
                try:
                    with Image.open(file_path) as img:
                        shapes.append(img.size)  # img.size returns (width, height)
                except Exception as e:
                    print(f"Could not open {file_path}: {e}")
        class_shapes[class_name] = shapes
    else:
        class_shapes[class_name] = []
        print(f"Directory not found: {class_path}")  # Debug print

# Print the shape of images in each class
for class_name, shapes in class_shapes.items():
    print(f"Class {class_name}: {len(shapes)} images with shapes {shapes[:5]} ...")  


# Initialize empty lists for images and labels
images = []
labels = []

# Load images and assign labels
for label, class_name in enumerate(class_names):
    class_path = os.path.join(resized_dir, class_name)
    image_files = os.listdir(class_path)

    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read as GBR
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img / 255.0  # Normalize pixel values (assuming 8-bit images)
        images.append(img)
        labels.append(label)

images, labels = shuffle(images, labels, random_state=42)

print('Data length:', len(images))
print('labels counts:', Counter(labels))

images = np.array(images).reshape(-1, 128, 128, 3)
labels = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5)
print('Train length:', len(X_train), X_train.shape)
print('Valid length:', len(X_test), X_val.shape)
print('Test length:', len(X_val), X_test.shape)

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)

base_model.trainable = False

NUM_CLASSES = len(labels)
Model_1 = Sequential()
Model_1.add(base_model)
Model_1.add(MaxPooling2D())
Model_1.add(layers.Flatten())
Model_1.add(layers.Dense(256, activation='relu'))
Model_1.add(layers.Dense(128, activation='relu'))
Model_1.add(layers.Dense(64, activation='relu'))
Model_1.add(layers.Dense(NUM_CLASSES, activation='softmax'))

optimizer=keras.optimizers.Adam(learning_rate=0.001)
losses=keras.losses.SparseCategoricalCrossentropy()
metrics=['accuracy']
Model_1.compile(optimizer=optimizer,loss=losses,metrics=metrics)



history_Model_1 = Model_1.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))







