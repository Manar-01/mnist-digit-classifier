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

## How to Run
1. Open the notebook:
``` bash
jupyter notebook qa_rag_climate_fever.ipynb
```
or use google colab 

2. Follow the steps in the notebook to:
  - Load and preprocess the dataset.
  - Build and train the CNN model.
  - Evaluate the model's performance.
  - Visualize prediction

## Dataset
The MNIST dataset is automatically loaded using the keras.datasets module. It contains:

- Training set: 60,000 images and labels.
- Test set: 10,000 images and labels.

Images are 28x28 grayscale, and labels represent digits from 0 to 9.

## Dependencies
- numpy
- matplotlib
- tensorflow (Keras is included in TensorFlow)
- scikit-learn
- jupyter

Install them using the requirements.txt file.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- The MNIST dataset was provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- This project uses the Keras and TensorFlow libraries for deep learning
