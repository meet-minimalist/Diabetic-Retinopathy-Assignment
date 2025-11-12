import pandas as pd
import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical

#Process training images
def preproc_dataset(dataset_path, image_height, image_width, is_alexnet=False):
    image_list = []
    y_labels = [] # List to store corresponding labels
    image_dir = dataset_path + '/train_images'
    labels_path = dataset_path + 'train.csv'
    labels_df = pd.read_csv(labels_path) # Replace with the actual path to your labels CSV file
    label_dict = labels_df.set_index('id_code')['diagnosis'].to_dict()
    for filename in os.listdir(image_dir)[:200]:
        # Extract image ID from filename (assuming filename format is 'image_id.jpg')
        image_id = os.path.splitext(filename)[0]
        if filename.endswith(('.jpg', '.png', '.jpeg')) and image_id in label_dict: # Add more image extensions if needed and check if label exists
            img_path = os.path.join(image_dir, filename)
            try:
                img = Image.open(img_path)
                img = img.resize((image_width, image_height))
                if is_alexnet:
                    img = img - np.mean(img)
                else:
                    img = np.array(img) / 255.0 # Normalize pixel values
                image_list.append(img)
                y_labels.append(label_dict[image_id]) # Get the label from the dictionary
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    all_images = np.array(image_list)
    all_labels = np.array(y_labels)
    all_labels = to_categorical(all_labels) #One hot encoding
    return all_images, all_labels