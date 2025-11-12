import pandas as pd
import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from config import imgnet_rgb_mean, imgnet_rgb_mean_resnet, imgnet_rgb_stddev_resnet


#Process training images
def preproc_dataset(dataset_path, image_height, image_width, model_type):
    image_list = []
    y_labels = [] # List to store corresponding labels
    image_dir = dataset_path + '/train_images'
    labels_path = dataset_path + 'train.csv'
    labels_df = pd.read_csv(labels_path) # Replace with the actual path to your labels CSV file
    label_dict = labels_df.set_index('id_code')['diagnosis'].to_dict()
    for filename in os.listdir(image_dir):
        # Extract image ID from filename (assuming filename format is 'image_id.jpg')
        image_id = os.path.splitext(filename)[0]
        if filename.endswith(('.jpg', '.png', '.jpeg')) and image_id in label_dict: # Add more image extensions if needed and check if label exists
            img_path = os.path.join(image_dir, filename)
            try:
                img = Image.open(img_path)
                img = img.resize((image_width, image_height))
                img = np.array(img).astype(np.float32)
                if model_type == "alexnet":
                    img = img - np.mean(img)
                elif model_type == "vgg":
                    img = img - np.array(imgnet_rgb_mean)
                elif model_type == "resnet":
                    img = np.array(img) / 255.0 # Normalize pixel values
                    img[..., 0] = (img[..., 0] - imgnet_rgb_mean_resnet[0]) / imgnet_rgb_stddev_resnet[0]
                    img[..., 1] = (img[..., 1] - imgnet_rgb_mean_resnet[1]) / imgnet_rgb_stddev_resnet[1]
                    img[..., 2] = (img[..., 2] - imgnet_rgb_mean_resnet[2]) / imgnet_rgb_stddev_resnet[2]
                else:
                    raise RuntimeError("Invalid model type. Choose from 'alexnet', 'vgg', or 'resnet'.")
                image_list.append(img)
                y_labels.append(label_dict[image_id]) # Get the label from the dictionary
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    all_images = np.array(image_list)
    all_labels = np.array(y_labels)
    all_labels = to_categorical(all_labels) #One hot encoding
    return all_images, all_labels