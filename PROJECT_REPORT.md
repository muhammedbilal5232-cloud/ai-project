# HANDWRITTEN DIGIT RECOGNITION SYSTEM
## Class XII – Artificial Intelligence Project

### 1. Abstract
The Handwritten Digit Recognition System is an Artificial Intelligence project that identifies handwritten numerical digits from 0 to 9. The user draws a digit on a digital canvas. The application processes the drawing and passes its pixel values to a trained Artificial Neural Network. The model predicts the digit and displays the result with probability estimates.

### 2. Introduction
Handwritten digit recognition is a common application of Artificial Intelligence and Machine Learning. It is used in areas such as reading postal codes, processing forms, and digitising handwritten records. Since handwriting differs from person to person, a computer must learn patterns from examples rather than depend only on fixed rules.

### 3. Aim
To develop an interactive Python application that recognises handwritten digits from 0 to 9 using a machine-learning model.

### 4. Objectives
- Understand a practical application of Artificial Intelligence.
- Learn how image data can be represented using pixel values.
- Train an Artificial Neural Network to classify digits.
- Build a simple user interface using Streamlit.
- Display a prediction and its estimated probability.

### 5. Tools and Technologies
- **Programming language:** Python
- **Interface:** Streamlit
- **Machine learning library:** Scikit-learn
- **Image processing:** Pillow (PIL)
- **Numerical operations:** NumPy
- **Model:** Multi-Layer Perceptron (Artificial Neural Network)

### 6. Dataset
The project uses the handwritten-digits dataset available in Scikit-learn. It contains labelled examples of digits from 0 to 9. Each image is represented as an 8 × 8 grid, giving 64 pixel values. The dataset is divided into training and testing portions. The training portion teaches the model, while the testing portion is used to estimate its performance on examples it did not train on.

### 7. Working Principle
1. The user draws a digit using the application canvas.
2. The drawing is converted into a grayscale image.
3. The image is resized to 8 × 8 pixels.
4. The pixel values are scaled to match the dataset format.
5. The trained neural network processes the 64 input values.
6. The model produces scores/probabilities for the ten digit classes.
7. The class with the highest probability is shown as the prediction.

### 8. Algorithm
1. Load the labelled handwritten-digits dataset.
2. Separate the input pixel values (X) and digit labels (y).
3. Split the dataset into training and testing sets.
4. Create an MLPClassifier neural network.
5. Train the model using the training data.
6. Evaluate it using the testing data.
7. Accept a new drawing from the user.
8. Preprocess the drawing and predict its digit.

### 9. Neural Network
A Multi-Layer Perceptron (MLP) is a type of Artificial Neural Network. It contains an input layer, one or more hidden layers, and an output layer. In this project, the input contains 64 pixel values and the output represents ten possible digit classes. During training, the model adjusts its internal weights to reduce prediction errors.

### 10. Expected Output
When a user draws a digit and selects **Predict digit**, the application displays:
- The predicted digit (0–9)
- The model's estimated confidence
- A chart showing the probability assigned to each digit

The model can make incorrect predictions. Its displayed confidence is an estimate and should not be treated as a guarantee.

### 11. Applications
- Reading digits on postal envelopes
- Digitising handwritten forms
- Processing numerical entries
- Supporting automated document processing
- Educational demonstrations of image classification

### 12. Advantages
- Easy-to-use drawing interface
- Fast predictions after the model is trained
- Demonstrates a real-world use of machine learning
- Uses freely available Python libraries and a built-in dataset

### 13. Limitations
- The model uses small 8 × 8 training images.
- Unusual handwriting may be misclassified.
- The drawing may look different from the examples in the dataset.
- The model recognises digits only, not letters or complete sentences.
- Prediction confidence does not guarantee correctness.

### 14. Future Scope
- Train with a larger dataset such as MNIST.
- Improve image-centering and preprocessing.
- Add support for recognising multiple digits.
- Develop a mobile-friendly version.
- Explore convolutional neural networks for image recognition.

### 15. Conclusion
The project demonstrates how Artificial Intelligence can be used to recognise handwritten digits. It combines image preprocessing, machine learning, and a simple graphical interface. The project helped explain how a computer learns from labelled examples and uses the learned patterns to make predictions.

### 16. Bibliography
1. Scikit-learn documentation – https://scikit-learn.org/
2. Streamlit documentation – https://docs.streamlit.io/
3. Python documentation – https://docs.python.org/3/
4. Pillow documentation – https://pillow.readthedocs.io/
