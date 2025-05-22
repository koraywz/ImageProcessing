Project Description
This project is an image classification system that uses Convolutional Neural Networks (CNNs) to classify images of cats and dogs. The primary goal is to build a model that can accurately distinguish between cat and dog images using deep learning techniques.

Dataset:
The dataset consists of cat and dog images stored under the PetImages/ directory. To ensure faster training and avoid memory issues, a maximum of 500 images per category were used.

Key Steps:
Image Preprocessing:

Resized all images to 100x100 pixels.

Converted BGR to RGB format.

Normalized pixel values to the [0,1] range.

Filtered out unreadable or corrupt images.

Data Augmentation:

Applied transformations such as rotation, zoom, horizontal flip, and shift to enrich the training set using ImageDataGenerator.

Model Architecture (CNN):

Multiple Conv2D layers with ReLU activation.

BatchNormalization and MaxPooling layers.

Fully connected layers with dropout regularization.

Final classification layer with softmax activation (2 classes: Cat, Dog).

Training & Validation:

Trained the model for 25 epochs with Adam optimizer and a learning rate of 0.0005.

Used 80% of the data for training/validation and 20% for testing.

Evaluation Metrics:

Accuracy Score on the test set.

Confusion Matrix for visualizing predictions.

ROC-AUC Score and ROC Curve to measure the model's discrimination ability.

Sample Prediction Visualization:

Displays a few test images with their predicted and actual labels.

Technologies Used:
Python, OpenCV, NumPy, Matplotlib, Seaborn

TensorFlow/Keras

Scikit-learn

Project Goal:
This project demonstrates how deep learning models like CNNs can be effectively used for binary image classification tasks. It serves as a foundation for more advanced applications in computer vision.
