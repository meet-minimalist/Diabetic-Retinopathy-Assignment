from sklearn.model_selection import train_test_split
from config import image_height, image_width, dataset_path, split_ratio, seed
from preproc_helper import preproc_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_test_data(model_type):
    all_images, all_labels = preproc_dataset(dataset_path, image_height, image_width, model_type)

    # Split the data (e.g., 80% train, 20% test split)
    # random_state is used for reproducibility
    x_train, x_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=split_ratio, random_state=seed
    )
    
    return x_train, x_test, y_train, y_test


def get_image_generator(model_type):
    x_train, x_test, y_train, y_test = get_train_test_data(model_type)
    
    # Create an ImageDataGenerator with desired augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=20,       # Rotate images by up to 20 degrees
        width_shift_range=0.1,   # Shift images horizontally by up to 10% of the width
        height_shift_range=0.1,  # Shift images vertically by up to 10% of the height
        shear_range=0.1,         # Apply shear transformation
        zoom_range=0.1,          # Zoom in or out by up to 10%
        horizontal_flip=True,    # Randomly flip images horizontall
    )

    # Fit the data generator on your training data
    datagen.fit(x_train)
    
    return x_train, y_train, x_test, y_test
