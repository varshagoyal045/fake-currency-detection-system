import numpy as np
from keras.models import load_model
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

# Load the trained model
model = load_model('models/model.h5')



def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image to the desired size
    image = cv2.resize(image, (224, 224))

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values
    image = image / 255.0

    return image


# def predict(image_path):
#     # Preprocess the input image
#     image = preprocess_image(image_path)

#     # Add a batch dimension to the image
#     image = np.expand_dims(image, axis=0)

#     # Make a prediction using the model
#     prediction = model.predict(image)

#     # Get the predicted class (0 or 1)
#     predicted_class = np.argmax(prediction)

#     return predicted_class


def predict(image_path):
    # Preprocess the input image
    image = preprocess_image(image_path)

    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)

    # Make a prediction using the model
    prediction = model.predict(image)

    # Get the predicted class (0 or 1)
    predicted_class = np.argmax(prediction)

    # Map the class index to the corresponding label
    class_labels = ['real', 'fake']
    predicted_label = class_labels[predicted_class]

    return predicted_label


def get_image_path():
    # Create a tkinter window
    window = tk.Tk()
    window.withdraw()

    # Open a file dialog to let the user select an image
    image_path = filedialog.askopenfilename()

    return image_path


def show_result(predicted_label):
    # Create a tkinter window
    window = tk.Tk()
    window.withdraw()

    # Display the predicted label in a dialog box
    messagebox.showinfo("Prediction Result",
                        f"Predicted label: {predicted_label}")


# Example usage:
image_path = get_image_path()


# # Example usage:
# # image_path = 'Dataset/Testing/Real.jpg'
# # image_path = 'Dataset/Validation/Real/39.jpg'
predicted_label = predict(image_path)
print(f'Predicted label: {predicted_label}')
show_result(predicted_label)



# # # Example usage:
# # image_path2 = 'Dataset/Validation/Fake/Fake.jpeg'
# # predicted_class = predict(image_path2)
# # print(f'Predicted class: {predicted_class}')

# import os
# print("TF_ENABLE_ONEDNN_OPTS =", os.getenv('TF_ENABLE_ONEDNN_OPTS'))




a = 4
print("value of a is " , a)
