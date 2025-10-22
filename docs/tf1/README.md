# TensorFlow Modules and Their Functions (Educational Focus)

TensorFlow is a modular deep learning framework consisting of many submodules that work together to build, train, and deploy machine learning models.  
This document focuses on **educational use cases** to help students understand how each module is used in practical learning examples.

---

## `tf.keras` — High-Level Neural Network API
**Purpose:** Simplifies creation, training, and evaluation of neural networks.

**Functions / Uses:**
- Layers (`Dense`, `Conv2D`, `LSTM`)
- Models (`Sequential`, `Model`)
- Training (`fit`, `evaluate`, `predict`)
- Optimizers, losses, and metrics

**Class Syntax Examples:**
```python
from tensorflow.keras.models import Sequential  # Import Sequential model class
from tensorflow.keras.layers import Dense, Conv2D, Flatten  # Common layer classes

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # Fully connected layer
    Dense(10, activation='softmax')  # Output layer for 10 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compile model
model.fit(x_train, y_train, epochs=5)  # Train model
```

**Educational Use Case:**  
- Build an MNIST digit classifier using `Sequential` and `Dense` layers.  
- Implement a simple CNN for image classification.

---

## `tf.data` — Input Pipeline
**Purpose:** Efficiently loads and preprocesses data for training.

**Functions / Uses:**
- Create pipelines from files, arrays, or generators.
- Transform (shuffle, batch, map, prefetch).

**Class Syntax Examples:**
```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # Create dataset from tensors
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)  # Optimize pipeline
```

**Educational Use Case:**  
- Load and preprocess images from folders for a classification task.  
- Use `.map()` to normalize images between 0–1.

---

## `tf.nn` — Neural Network Operations
**Purpose:** Core neural network building blocks.

**Functions / Uses:**
- Activations: `relu`, `sigmoid`
- Convolutions: `conv2d`
- Pooling, normalization

**Class Syntax Examples:**
```python
import tensorflow as tf

x = tf.random.normal([1, 28, 28, 3])  # Input tensor
filters = tf.random.normal([3, 3, 3, 16])  # Filter weights

y = tf.nn.conv2d(x, filters, strides=1, padding='SAME')  # Perform 2D convolution
y = tf.nn.relu(y)  # Apply ReLU activation
y = tf.nn.max_pool2d(y, ksize=2, strides=2, padding='SAME')  # Max pooling layer
```

**Educational Use Case:**  
- Implement a perceptron manually using `tf.nn.relu`.  
- Visualize the effect of different activation functions.

---

## `tf.train` — Training Utilities (Low-Level)
**Purpose:** Tools for training process control.

**Functions / Uses:**
- Gradient computation and checkpointing.

**Class Syntax Examples:**
```python
import tensorflow as tf

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)  # Define optimizer
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)  # Manage checkpoints

# Custom training loop
with tf.GradientTape() as tape:  # Record gradients
    predictions = model(x_batch)
    loss = tf.reduce_mean(tf.square(y_batch - predictions))  # Compute loss
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))  # Update weights
```

**Educational Use Case:**  
- Create a custom training loop using `tf.GradientTape`.  
- Save and restore model weights during training.

---

## `tf.optimizers` — Optimization Algorithms
**Purpose:** Provides gradient-based optimization algorithms.

**Functions / Uses:**
- `Adam`, `SGD`, `RMSprop`

**Class Syntax Examples:**
```python
from tensorflow.keras import optimizers

opt = optimizers.Adam(learning_rate=0.001)  # Adam optimizer
opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)  # SGD optimizer
```

**Educational Use Case:**  
- Compare optimization performance between SGD and Adam.  
- Plot convergence curves for different optimizers.

---

## `tf.losses` — Loss Functions
**Purpose:** Measures how well predictions match actual values.

**Functions / Uses:**
- Regression: `MeanSquaredError`
- Classification: `BinaryCrossentropy`, `CategoricalCrossentropy`

**Class Syntax Examples:**
```python
from tensorflow.keras import losses

mse = losses.MeanSquaredError()  # Mean squared error for regression
cross_entropy = losses.CategoricalCrossentropy()  # Cross-entropy for classification

y_true = [[0, 1, 0]]
y_pred = [[0.2, 0.7, 0.1]]
print(cross_entropy(y_true, y_pred).numpy())  # Compute loss value
```

**Educational Use Case:**  
- Demonstrate how changing the loss affects model learning.  
- Implement MSE loss manually for linear regression.

---

## `tf.summary` — Visualization (TensorBoard)
**Purpose:** Logs metrics and graphs for visualization in TensorBoard.

**Educational Use Case:**  
- Track model accuracy and loss across epochs.  
- Visualize computational graphs of student models.

---
## `tf.saved_model` — Model Saving and Loading
**Purpose:** Export/import models for reuse or deployment.

**Educational Use Case:**  
- Save trained models and reload for inference demonstrations.  
- Share models among students for collaborative projects.

---

## `tf.image` — Image Processing
**Purpose:** Image augmentation and transformation.

**Functions / Uses:**
- Resize, crop, flip, adjust brightness/contrast.

**Educational Use Case:**  
- Apply data augmentation on CIFAR-10 dataset.  
- Show how image transformations affect CNN learning.

---

## 12. `tf.io` — Input/Output Operations
**Purpose:** Reading/writing files and parsing data formats.

**Educational Use Case:**  
- Read datasets from CSV files for regression or classification.  
- Save model checkpoints to a directory.

---
## `tf.hub` — Pretrained Models
**Purpose:** Access reusable pretrained models.

**Educational Use Case:**  
- Use a pretrained MobileNet for transfer learning.  
- Fine-tune a text embedding model for classification tasks.

---

## `tf.lite` — TensorFlow Lite
**Purpose:** Converts models for mobile and embedded devices.

**Educational Use Case:**  
- Convert MNIST model to TFLite and run on Raspberry Pi.  
- Show how quantization reduces model size.

---

## `tf.js` — TensorFlow.js
**Purpose:** ML in web browsers using JavaScript.

**Educational Use Case:**  
- Deploy a digit classifier in a browser demo.  
- Let students experiment with real-time webcam classification.

---



## `tf.distribute` — Distributed Training
**Purpose:** Parallelize model training on GPUs or clusters.

**Educational Use Case:**  
- Demonstrate multi-GPU training in a lab setting.  
- Discuss how distributed strategies speed up training.

---