# SVHN Classification Using CNN

This repository contains code for training a Convolutional Neural Network (CNN) model to classify images from the [SVHN dataset](http://ufldl.stanford.edu/housenumbers/), a collection of street-view house numbers. The model is built and trained using TensorFlow and Keras.

## Dataset

- **Train Dataset**: `train_32x32.mat` contains 32x32 color images of digits (from 0 to 9).
- **Test Dataset**: `test_32x32.mat` contains similar images for testing.
- Each image has three channels (RGB), and the labels represent the digit class (0-9).

## Steps Involved

### 1. **Loading the Dataset**

The dataset is loaded using the `scipy.io.loadmat` function to import `.mat` files. The images and labels are preprocessed:
- Transposed to have dimensions `(number of images, height, width, channels)`.
- Labels are one-hot encoded for classification.

### 2. **Data Normalization**

All pixel values are scaled to the range `[0, 1]` by dividing by `255.0`.

### 3. **Data Augmentation**

To prevent overfitting, several augmentation techniques are applied to the training dataset:
- Random horizontal flip
- Random brightness adjustment
- Random saturation adjustment

These augmentations are applied during training to artificially expand the dataset.

### 4. **Splitting the Dataset**

The training data is split into training and validation sets using an 80-20 split. The test dataset remains unchanged for model evaluation.

### 5. **Building the CNN Model**

A Convolutional Neural Network is built using TensorFlow’s `Sequential` API. The architecture consists of:
- 3 Convolutional layers with ReLU activation and MaxPooling.
- A Flatten layer to prepare for fully connected layers.
- A Dense layer with 128 neurons and ReLU activation.
- A Dropout layer for regularization (0.5 dropout rate).
- A final Dense layer with 10 neurons for classification (softmax activation).

### 6. **Model Compilation**

The model is compiled with the following settings:
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy

### 7. **Training the Model**

The model is trained on the augmented training dataset for 20 epochs. During training, validation data is used to monitor performance and avoid overfitting.

### 8. **Model Evaluation**

After training, the model is evaluated on the test dataset (which was not augmented). The test accuracy and loss are computed.

### 9. **Plotting the Confusion Matrix**

To visualize the performance of the model, a confusion matrix is plotted. This matrix shows the correct and incorrect predictions for each digit class.

