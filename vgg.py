
# Step 2: Implement VGG-16 baseline

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet18
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam # Import the Adam optimizer
from sklearn.metrics import classification_report # Import classification_report
import numpy as np # Import numpy


def vgg_baseline(image_height, image_width, num_classes=5):
    base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

    # Freeze the weights of the pre-trained layers
    for layer in base_model_vgg.layers:
        layer.trainable = False

    # Add a new classification head for 5 classes
    last_layer = Flatten()(base_model_vgg.output)
    last_layer = Dense(512, activation='relu')(last_layer) # Add a Dense layer before the output layer
    last_layer = Dense(256, activation='relu')(last_layer) # Add a Dense layer before the output layer
    last_layer = Dense(num_classes, activation='softmax')(last_layer) # Final Dense layer for 5 classes with softmax activation

    # Create the new model
    model_vgg = Model(inputs=base_model_vgg.input, outputs=last_layer)

    return model_vgg

