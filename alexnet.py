import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np
from config import alexnet_weights_path, num_classes
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam # Import the Adam optimizer
from sklearn.metrics import classification_report # Import classification_report
import numpy as np # Import numpy



class LRN(layers.Layer):
    """Local Response Normalization as a custom layer"""
    def __init__(self, **kwargs):
        super(LRN, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.local_response_normalization(
            inputs, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0
        )

    def get_config(self):
        return super(LRN, self).get_config()

def create_alexnet_sequential(image_height, image_width, num_classes=1000, include_top=True):
    """Create AlexNet using Sequential API"""

    model = Sequential([
        # Conv Block 1
        layers.Conv2D(96, 11, strides=4, padding='same', activation='relu',
                     input_shape=(image_height, image_width, 3), name='conv1'),
        LRN(name='norm1'),
        layers.MaxPooling2D(3, strides=2, padding='valid', name='pool1'),

        # Conv Block 2
        layers.Conv2D(256, 5, strides=1, padding='same', activation='relu', groups=2, name='conv2'),
        LRN(name='norm2'),
        layers.MaxPooling2D(3, strides=2, padding='valid', name='pool2'),

        # Conv Block 3-5
        layers.Conv2D(384, 3, strides=1, padding='same', activation='relu', name='conv3'),
        layers.Conv2D(384, 3, strides=1, padding='same', activation='relu', groups=2, name='conv4'),
        layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', groups=2, name='conv5'),
        layers.MaxPooling2D(3, strides=2, padding='valid', name='pool5'),
    ])
    
    if include_top:
        # Fully Connected Layers
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu', name='fc6'))
        model.add(layers.Dense(4096, activation='relu', name='fc7'))
        model.add(layers.Dense(num_classes, activation='softmax', name='fc8'))
    return model


def load_pretrained_weights(model, npy_file_path, include_top=True):
    """Load pretrained weights from .npy file"""
    net_data = np.load(npy_file_path, allow_pickle=True, encoding='latin1').item()

    # Set weights for each layer by name
    weight_mapping = {
        'conv1': 'conv1',
        'conv2': 'conv2',
        'conv3': 'conv3',
        'conv4': 'conv4',
        'conv5': 'conv5',
    }
     
    if include_top:
        weight_mapping.update({
            'fc6': 'fc6',
            'fc7': 'fc7',
            'fc8': 'fc8'
        })

    for layer in model.layers:
        if layer.name in weight_mapping:
            npy_key = weight_mapping[layer.name]
            if npy_key in net_data:
                layer.set_weights([net_data[npy_key][0], net_data[npy_key][1]])
                print(f"✓ Loaded weights for {layer.name}")

    return model


# Create and load the model
def create_alexnet_with_pretrained_weights(npy_file_path, image_height, image_width, num_classes=1000, include_top=True):
    model = create_alexnet_sequential(image_height, image_width, num_classes=num_classes, include_top=include_top)
    model = load_pretrained_weights(model, npy_file_path, include_top=include_top)
    return model

def alexnet_baseline(image_height, image_width, num_classes):
    base_model_alexnet = create_alexnet_with_pretrained_weights(alexnet_weights_path, image_height, image_width, num_classes=num_classes, include_top=False)
    
    # Freeze the weights of the pre-trained layers
    for layer in base_model_alexnet.layers:
        layer.trainable = False

    model_alexnet = Sequential([
        base_model_alexnet,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax'),
    ])
    
    return model_alexnet