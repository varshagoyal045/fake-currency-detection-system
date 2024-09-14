import os
import numpy as np
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical


# Load the preprocessed dataset
training_dataset = np.load('training_dataset.npy')
training_labels = np.load('training_labels.npy')

testing_dataset = np.load('testing_dataset.npy')
testing_labels = np.load('testing_labels.npy')

# One-hot encode the labels
training_labels = to_categorical(training_labels)
testing_labels = to_categorical(testing_labels)

print("!"*100)

# Define the CNN model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(training_dataset, training_labels, epochs=10, batch_size=32,
          validation_data=(testing_dataset, testing_labels))

# model score
loss, accuracy = model.evaluate(testing_dataset, testing_labels)
print(f'Test accuracy: {accuracy:.2f}')




# Create the models folder if it doesn't exist
models_folder = 'models'
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

# Save the model in the models folder
model_path = os.path.join(models_folder, 'model.h5')
model.save(model_path)
