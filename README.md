# MNIST Digit Classifier

This project builds a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the MNIST dataset. Using Python libraries such as Keras, Scikit-learn, NumPy, and Matplotlib, this project demonstrates a complete machine learning pipeline for image classification.

---

## Features

- **Dataset**: MNIST dataset with 70,000 grayscale images of size 28x28 pixels.
- **Preprocessing**:
  - Normalize pixel values to range [0, 1].
  - Encode labels for multi-class classification using to_categorical function.
  - Split data into training and testing subsets.
- **Model**:
  - CNN architecture with:
    - Convolutional layers for feature extraction.
    - Max-pooling layers for downsampling.
    - Dense layers for classification.
  - Trained using the Adam optimizer and categorical cross-entropy loss.
- **Evaluation**:
  - Accuracy and loss metrics.
  - Visualize predictions and results using Matplotlib.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/mnist-digit-classifier.git
cd mnist-digit-classifier
```
## Install Dependencies
Ensure you have Python 3.7+ and install required packages:
```bash
pip install -r requirements.txt
```
## Acknowledgments
- The MNIST dataset was provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- This project uses the Keras and TensorFlow libraries for deep learning
