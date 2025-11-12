# Step 2: Implement VGG-16 baseline

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam # Import the Adam optimizer
from sklearn.metrics import classification_report # Import classification_report
import numpy as np # Import numpy
from config import image_height, image_width, num_classes, num_epochs, batch_size, lr


def train_model(model_name):
    if model_name == "vgg":
        from vgg import vgg_baseline
        model = vgg_baseline(image_height, image_width, num_classes)
    elif model_name == "resnet":
        from resnet import resnet_baseline
        model = resnet_baseline(image_height, image_width, num_classes)
    elif model_name == "alexnet":
        from alexnet import alexnet_baseline
        model = alexnet_baseline(image_height, image_width, num_classes)
    else:
        raise RuntimeError("Invalid model name. Choose from 'vgg', 'resnet', or 'alexnet'.")

    # This will show device placement
    tf.debugging.set_log_device_placement(True)

    # Compile the model (only the new head's weights will be updated)
    model.compile(optimizer=Adam(learning_rate=lr), # Use the defined optimizer
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    from dataset_helper import get_image_generator
    x_train, y_train, x_test, y_test = get_image_generator(model_type=model_name)


    # Train the VGG model (only the newly added classification head)
    # Use the train_dataset
    history_vgg = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, # Increased batch size to 64
                            validation_data=(x_test, y_test)) # Use test_dataset for evaluation

    print("Training completed. Evaluating on test data...")

    # Evaluate the VGG model on the test data
    # Use the test_dataset
    loss, acc = model.evaluate(x_test, y_test, verbose=2)

    print(f"Test Accuracy: {acc * 100:.2f}%")

    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1) # Convert probabilities to class predictions

    # Convert one-hot encoded y_test back to integer labels for classification_report
    y_test_labels = np.argmax(y_test, axis=1)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred))
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model on the dataset.")
    parser.add_argument('--model', type=str, required=True,
                        help="Model to train: 'vgg', 'resnet', or 'alexnet'")
    args = parser.parse_args()
    train_model(args.model)
    