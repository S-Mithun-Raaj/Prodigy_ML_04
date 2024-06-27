# Prodigy_ML_04
Fourth Task for ML internship in Prodigy infotech
To create a README file for your gesture recognition project, follow these guidelines to provide clear instructions and information about your project:

---

# Hand Gesture Recognition using Deep Learning

## Overview

This project implements a hand gesture recognition system using convolutional neural networks (CNNs). The model is trained on a dataset consisting of near-infrared images acquired by the Leap Motion sensor. It recognizes and classifies hand gestures performed by different subjects.

## Dataset

The dataset used for training and testing contains:
- **Subjects:** 10 individuals (5 men, 5 women)
- **Gestures:** 10 different hand gestures including palm, fist, thumb, index finger, etc.
- **Structure:** Each subject has a folder with subfolders for each gesture containing image samples.

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog).

## Project Structure

The project is organized as follows:
- **Training Data:** Located at `C:\Users\mithu\Desktop\task4\train`
- **Testing Data:** Located at `C:\Users\mithu\Desktop\task4\test`
- **Model:** The trained model is saved as `hand_gesture_recognition_model.keras` in the project directory.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy

## Setup Instructions

1. **Clone the repository:**
   ```bash
 (https://github.com/S-Mithun-Raaj/Prodigy_ML_04)  ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare the dataset:**
   - Download the dataset from the provided Kaggle link.
   - Extract the dataset and place it in appropriate directories (`train` and `test`).

4. **Training the model:**
   - Run the `split_dataset.py` script to split the dataset into training and testing sets.
   - Execute `train_model.py` to train the CNN model using the training data.

5. **Testing the model:**
   - After training, use `gesture_recognition_.py` to predict hand gestures from images in the `test` folder.

## Usage

- Ensure Python environment is set up correctly with dependencies installed.
- Run the scripts in the specified order to train and test the model.
- Adjust parameters and paths in scripts as needed for your environment.

## Notes

- The model's performance can be further improved by tuning hyperparameters or using more advanced architectures.
- Experiment with different image preprocessing techniques to enhance accuracy.

## Credits

- Dataset: T. Mantecón, C.R. del Blanco, F. Jaureguizar, N. García, “Hand Gesture Recognition using Infrared Imagery Provided by Leap Motion Controller”, Int. Conf. on Advanced Concepts for Intelligent Vision Systems, ACIVS 2016.

---

Adjust the paths, URLs, and specific details according to your project setup and requirements. This README provides a structured overview of your project, making it easier for others (and yourself) to understand and use your hand gesture recognition system effectively.
