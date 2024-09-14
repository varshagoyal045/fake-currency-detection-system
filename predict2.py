import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk

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

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("400x400")

        # Create a frame for the image preview
        self.preview_frame = tk.Frame(self.root, bg="gray")
        self.preview_frame.pack(fill="both", expand=True)

        # Create a label for the image preview
        self.preview_label = tk.Label(self.preview_frame, text="Select an image", bg="gray")
        self.preview_label.pack(fill="both", expand=True)

        # Create a button to select an image
        self.select_button = tk.Button(self.root, text="Select Image", command=self.get_image_path, bg="blue", fg="white")
        self.select_button.pack(fill="x")

        # Create a button to classify the image
        self.classify_button = tk.Button(self.root, text="Classify", command=self.classify_image, bg="green", fg="white")
        self.classify_button.pack(fill="x")

        # Create a label to display the result
        self.result_label = tk.Label(self.root, text="", bg="white")
        self.result_label.pack(fill="x")

        # Create a label to display the confidence level
        self.confidence_label = tk.Label(self.root, text="", bg="white")
        self.confidence_label.pack(fill="x")

        # Initialize the image path
        self.image_path = None
        self.image_preview = None

    def get_image_path(self):
        # Open a file dialog to let the user select an image
        self.image_path = filedialog.askopenfilename()

        # Display the selected image in the preview frame
        image = cv2.imread(self.image_path)
        image = cv2.resize(image, (200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self.image_preview = ImageTk.PhotoImage(image)
        self.preview_label.config(image=self.image_preview)
        self.preview_label.image = self.image_preview

    def classify_image(self):
        if self.image_path is None:
            messagebox.showerror("Error", "Please select an image first")
            return

        # Preprocess the input image
        image = preprocess_image(self.image_path)

        # Add a batch dimension to the image
        image = np.expand_dims(image, axis=0)

        # Make a prediction using the model
        prediction = model.predict(image)

        # Get the predicted class (0 or 1)
        predicted_class = np.argmax(prediction)

        # Map the class index to the corresponding label
        class_labels = ['real', 'fake']
        predicted_label = class_labels[predicted_class]

        # Calculate the confidence level
        confidence = prediction[0][predicted_class]

        # Display the result in the result label
        self.result_label.config(text=f"Predicted label: {predicted_label}")

        # Display the confidence level in the confidence label
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()