import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to preprocess an image file for prediction
def preprocess_image(file_path, target_size=(128, 128)):
    print(f"Preprocessing image: {file_path}")
    img = load_img(file_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit batch size
    img_array /= 255.0  # Normalize pixel values
    return img_array, img

# Load the saved model
model_path = r'C:\Users\mithu\Desktop\task4\hand_gesture_recognition_model.keras'
print(f"Loading model from: {model_path}")
model = load_model(model_path)
print("Model loaded successfully.")

# Path to your test data directory
test_data_dir = r'C:\Users\mithu\Desktop\task4\test'

# Assuming that the training directory is known to get the class labels
train_data_dir = r'C:\Users\mithu\Desktop\task4\train'

# Get the class labels from the training directory
class_labels = sorted(os.listdir(train_data_dir))
print(f"Class labels: {class_labels}")

# Function to gather one image file from each folder in the test directory
def gather_one_image_per_folder(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            # Look for image files in the current folder
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(os.path.join(folder_path, file))
                    break  # Take only the first image from each folder
    return image_files

# Get list of one image per folder from the test data directory
test_data_files = gather_one_image_per_folder(test_data_dir)
print(f"Found {len(test_data_files)} test files.")

# Function to predict and display results
def predict_and_display(file_path):
    try:
        # Preprocess the image
        img_array, img = preprocess_image(file_path)
        
        # Predict the class probabilities
        predictions = model.predict(img_array)
        
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)
        
        # Get the predicted class label
        predicted_class = class_labels[predicted_class_index]
        
        # Display the prediction result in the terminal
        print(f"Image: {file_path} - Predicted Class: {predicted_class}")
        
        # Display the image with the prediction
        fig, ax = plt.subplots()
        ax.imshow(img_array[0])  # Display the preprocessed image array
        ax.set_title(f"Predicted Class: {predicted_class}")
        ax.axis('off')
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Start predicting and displaying images
print("Starting predictions...")

# Display one image from each folder sequentially
for file_path in test_data_files:
    predict_and_display(file_path)
