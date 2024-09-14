import os
import cv2
import numpy as np

# Define the image size and pixel value range
IMAGE_SIZE = (224, 224)
PIXEL_VALUE_RANGE = (0, 255)

# Define the dataset folders
DATASET_FOLDER = 'Dataset'
TRAINING_FOLDER = 'Training'
TESTING_FOLDER = 'Validation'
REAL_FOLDER = 'Real'
FAKE_FOLDER = 'Fake'

# Define the output dataset files
TRAINING_DATASET_FILE = 'training_dataset.npy'
TRAINING_LABELS_FILE = 'training_labels.npy'
TESTING_DATASET_FILE = 'testing_dataset.npy'
TESTING_LABELS_FILE = 'testing_labels.npy'


def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Resize the image to the desired size
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to the desired range
    image = image / PIXEL_VALUE_RANGE[1]

    return image


def create_dataset(folder, output_file_dataset, output_file_labels):
    dataset = []
    labels = []

    # Iterate over the Real and Fake folders
    for label, folder_name in enumerate([REAL_FOLDER, FAKE_FOLDER]):
        folder_path = os.path.join(folder, folder_name)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = preprocess_image(image_path)
            if image is not None:
                dataset.append(image)
                labels.append(label)

    # Save the dataset and labels to separate files
    dataset = [img.reshape((IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
               for img in dataset]
    np.save(output_file_dataset, dataset)
    np.save(output_file_labels, labels)


# Create the training dataset
create_dataset(os.path.join(DATASET_FOLDER, TRAINING_FOLDER),
               TRAINING_DATASET_FILE, TRAINING_LABELS_FILE)

# Create the testing dataset
create_dataset(os.path.join(DATASET_FOLDER, TESTING_FOLDER),
               TESTING_DATASET_FILE, TESTING_LABELS_FILE)
