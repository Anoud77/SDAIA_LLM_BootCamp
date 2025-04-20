# Deep Learning Projects from SDAIA LLM Bootcamp

This document outlines three deep learning projects from the SDAIA LLM bootcamp, covering fundamental concepts and practical applications using TensorFlow and Keras.

## Project 1: Deep Learning Models with Keras

**Objective:** Build artificial neural networks (ANNs) for deep learning using the Keras library with TensorFlow 2.0. Explore building and training models with the Keras Sequential and Functional APIs.

### 1.  Defining a Sequential Model

* The Keras Sequential model is used for building simple neural networks where layers are stacked sequentially.
* The code defines a model with three dense (fully connected) layers:
    * Layer 1: 4 neurons, ReLU activation, input dimension of 2.
    * Layer 2: 6 neurons, ReLU activation.
    * Layer 3: 1 neuron, sigmoid activation (for binary classification).
* `model.summary()` displays the model architecture, output shapes, and the number of parameters.
* The code also provides a brief explanation of the ReLU and sigmoid activation functions with source URLs.
* `plot_model` visualizes the network architecture.

### 2.  Functional API

* The Functional API offers more flexibility than the Sequential model, allowing for complex layer interactions.
* The code defines a model with four dense layers:
    * Input layer with a shape of (10,).
    * Three hidden layers (10, 20, and 10 neurons) with ReLU activation.
    * Output layer with 1 neuron and sigmoid activation.
* Layers are connected by passing the output of one layer to the next within parentheses (e.g., `Dense(10, activation='relu')(visible)`).
* The model is defined using `Model(inputs=visible, outputs=layer4)`.
* `model.summary()` and `plot_model` are used to display the model structure.

### 3.  Model Training

* This section describes the process of building, compiling, and training a Keras model.
* A Sequential model is built with two dense layers (32 neurons with ReLU and 1 neuron with sigmoid activation).
* The model is compiled with the stochastic gradient descent (`sgd`) optimizer, `binary_crossentropy` loss function, and `accuracy` metric.
* Dummy data (randomly generated) is used to demonstrate the training process.
* The model is trained using `model.fit()` with a batch size of 32 and 10 epochs.
* The trained model is used to make a prediction on new dummy data using `model.predict()`.

### 4.  Custom Model

* This exercise involves building a custom model using both the Sequential and Functional APIs.
* **Sequential API:** A simple three-layer network is defined.
* **Functional API:** A similar three-layer network is defined, demonstrating the more explicit layer connections of the Functional API.

## Project 2: Binary and Multi-class Classification

**Objective:** This project contains two exercises: multi-class classification with the Iris flower dataset and binary classification with the sonar dataset.

### 1.  Multi-class Classification with Iris Dataset

* **Load Data:** The Iris dataset is loaded using pandas, and the input features (X) and output variable (Y) are separated.
* **Encode Output Variable:**
    * The categorical output variable (flower species) is encoded as integers using `LabelEncoder`.
    * The integer encoding is then converted to a one-hot encoded representation using `to_categorical`.
* **Define Keras Model:**
    * A function `create_model()` is defined to create a Keras Sequential model.
    * The model has one hidden layer with 10 neurons and ReLU activation, and an output layer with 3 neurons (for the three classes) and softmax activation.
    * The model is compiled with the `sgd` optimizer, `categorical_crossentropy` loss, and `accuracy` metric.
* **Train Model:** The model is trained on the Iris dataset for 20 epochs with a batch size of 5.

### 2.  Binary Classification with Sonar Dataset

* **Load Dataset:** The Sonar dataset is loaded, and input features (X1) and the output variable (Y) are separated.
* **Encode Output Variable:** The categorical output variable ('M' for mine, 'R' for rock) is encoded as integers (1 and 0) using `LabelEncoder`.
* **Define Keras Model:**
    * A function `create_baseline()` is defined to create a Keras Sequential model for binary classification.
    * The model has one hidden layer with 60 neurons, 'normal' weight initialization, and ReLU activation, and one output layer with 1 neuron, 'normal' weight initialization, and sigmoid activation.
    * The model is compiled with the `adam` optimizer, `binary_crossentropy` loss, and `accuracy` metric.
* **Evaluate Model:**
    * The model's performance is evaluated using stratified k-fold cross-validation (10 splits) with `KerasClassifier` from `tensorflow.keras.wrappers.scikit_learn`.
    * The mean and standard deviation of the cross-validation results are printed.
* **Apply Standardization on Dataset:**
    * The `StandardScaler` is used to standardize the Sonar dataset (rescale features to have mean 0 and standard deviation 1).
* **Create a Pipeline:**
    * A pipeline is created using `Pipeline` to combine the `StandardScaler` and the Keras model. This ensures that the data is standardized before being fed to the model during cross-validation.
* **Evaluate Model:** The model is evaluated again using stratified k-fold cross-validation with the pipeline, and the results are printed.

## Project 3: Stroke Prediction

**Objective:** Develop a solution to predict whether a person will have a stroke using deep learning.

### 1.  Importing Libraries

* The code imports `warnings`, `numpy`, `pandas`, and `matplotlib.pyplot`. `warnings.filterwarnings('ignore')` is used to suppress warnings.

### 2.  Loading the Dataset

* The stroke dataset is loaded from a CSV file (`healthcare-dataset-stroke-data.csv`) using pandas. The first few rows of the dataset are displayed using `data.head()`.

### 3.  Exploratory Data Analysis

* **Shape of the Data:** The number of rows and columns in the dataset is printed using `data.shape`.
* **Types of Different Columns:** The data types of each feature are printed using `data.dtypes`.
* **Dealing with Categorical Variables:** The code uses `.value_counts()` to display the unique values and their counts for the following categorical features: 'gender', 'Residence_type', 'work_type', 'ever_married', 'hypertension', 'heart_disease', and 'stroke'.
* **Dealing with Nulls:** The 'bmi' column, which contains null values, is filled with the mean value of 'bmi' using `data['bmi'].fillna()`.
* **Encoding Categorical Features:**
    * The following categorical features are encoded into numerical values using `LabelEncoder`: 'smoking_status', 'Residence_type', 'work_type', 'ever_married', and 'gender'.

### 4.  Preprocessing

* **Normalizing Features:** The input data is normalized by dividing all values by the maximum value in each column. This scales the features to the range \[0, 1\]. The normalized data is described using `data.describe()`.
* **Removing Unnecessary Features:** The 'id' column is dropped as it is irrelevant for prediction.

### 5.  Building the DL Model

* A Sequential model is built with the following architecture:
    * Layer 1: 64 neurons, ReLU activation, input dimension of 10.
    * Layer 2: 32 neurons, ReLU activation.
    * Layer 3: 16 neurons, ReLU activation.
    * Layer 4: 1 neuron, sigmoid activation (for binary classification).
* `model.summary()` displays the model architecture.

### 6.  Compiling the Model

* The model is compiled with the `adam` optimizer, `binary_crossentropy` loss, and the metrics 'accuracy', 'Precision', and 'Recall'.

### 7.  Fitting the Model

* The data is split into training (70%) and validation (30%) sets using `train_test_split`, stratifying by the target variable ('stroke') to maintain class balance.
* The model is trained on the training data for 15 epochs, with the validation data used to monitor performance during training.

### 8.  Improving DL Models

* **Checking For Data Imbalance:** The code identifies a class imbalance problem (likely many more non-stroke cases than stroke cases), which can lead to poor precision and recall.
* **Oversampling:** The SMOTE (Synthetic Minority Over-sampling Technique) is used to oversample the minority class (stroke cases) in the training data. SMOTE generates synthetic samples to balance the class distribution.
* The balanced dataset is split into training, validation, and test sets (80%, 20%, and 10%, respectively).
* The model is trained on the balanced training data, and its performance is evaluated on the validation set.
