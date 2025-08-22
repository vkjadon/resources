# Machine Learning (ML)
**Machine Learning** = The practice of using an algorithm to analyse data, learn from that data and make predictions about new data.

## ML and Traditional Programming

Traditional programming is a manual process — meaning a programmer creates the program and formulate the rules for the logics whereas the machine learning is an automated process and the algorithm automatically formulates the rules from the data.

## Deep Learning

Deep learning is a subfield of machine learning and artificial intelligence (AI) that aims to enable computers to learn and make intelligent decisions without explicitly programmed.

Deep learning algorithms are designed to automatically learn representations of data through multiple layers of interconnected nodes called neurons. The neurons process and transmit information to other interconnected neurons using mathematical functions and thus, also called as computational units. The network so produced are called artificial neural networks.

The term "deep" in deep learning refers to the presence of multiple layers in these neural networks. Each layer extracts and transforms the input data, gradually learning complex patterns and features of the data. The information flows through these layers in a hierarchical manner, with each layer building upon the representations learned by the previous layer.

This hierarchical feature extraction is what distinguishes deep learning from traditional machine learning approaches.

## Artificial Neural Network

Networks using Deep learning are called Artificial neural Networks(ANN).

**Artificial Neural Networks(ANN)**- ANN are computing systems that are inspired by the brain's neural networks.

These networks are based on a collection of connected units called **neurons** or **artificial neurons**.

Neurons are oganised in layers -

*   Input layer
*   Hidden layers
*   Output layer

**Layers in an ANN**
ANN are typically organised in layers.Different types of layers include



*   Dense (or fully connected layers)
*   Convolutional Layers
*   Pooling layers
*   Recurrent layers
*   Normalization layers
*   Many others

output= input*weight

`weight` is assigned to every input-output connection between neurons.

## Activation Function

Activation Function defines output of a neuron given a set of inputs.

Output = Activation(weighted sum of inputs)

**Sigmoid Activation Function** transforms the negative input to a number very close to zero.If input is a positive number then it transforms the input to a number very close to 1 and if input is close to zero then it will be transformed to a number between 0 and 1.

It means that lower limit for this activation function is 0 and 1 will be upper limit.

**Rectified Linear Unit(ReLu)** if the input is less than zero than relu will transform the input to zeroand if the number is greater than zero,relu will give output the input number.
The more positive the neuron(value), the more activated it will be.


## Parameters and Hyperparameters
Hyperparameters are those parameters that either modify our network structure or affect our training process, but are not a part of the network parameters
learned during training. Examples include; the learning rate, the number of layers, the number of units per layer, and the activation function type.

## Input Data
The significance of the input data in deep learning cannot be overstated. In fact, data is the lifeblood of deep learning models. It influences the model's performance, generalization ability, robustness, and its capability to extract meaningful representations.
Therefore, obtaining high-quality, diverse, and well-curated data is essential for achieving optimal results in deep learning applications. Here are several reasons why data plays a critical role:

## Model Training
Deep learning models learn patterns and make predictions based on the data they are trained on. The quality, quantity, and diversity of the data used for training directly impact the performance and generalization ability of the model. A large and representative dataset allows the model to learn robust and accurate representations.

## Feature Learning
Deep learning models have the remarkable ability to learn hierarchical representations from raw data. By exposing the model to diverse and informative data, it can automatically extract relevant features and representations, enabling it to make better predictions or classifications.

## Generalization

The success of a deep learning model lies in its ability to generalize well to unseen data. Training a model on a diverse dataset helps it learn patterns and variations present in the data, making it more likely to perform well on new, unseen instances.

## Transfer Learning
Pretrained deep learning models can be used as a starting point for new tasks by fine-tuning them on task-specific datasets. Access to large and diverse datasets allows for effective transfer learning, where the model can leverage previously learned representations and adapt them to new domains.

## Acquiring Data

**Define Your Data Needs**

Start by clearly defining the purpose of your machine learning project and the specific data requirements. Consider the type of data (structured or unstructured), the features needed, and the target variable or outcome you want to predict.

**Identify Data Sources**

Identify potential sources of data that align with your project requirements. These sources can include public datasets, proprietary databases, APIs, sensor data, web scraping, social media platforms, surveys, or data collected internally within your organization.

**Data Collection**

Collect the data from the identified sources. The collection process may vary depending on the data source. It could involve downloading datasets, querying databases, using web scraping tools, or setting up data collection pipelines for real-time streaming data. Ensure that you comply with legal and ethical considerations and obtain necessary permissions when collecting data.

**Data Preprocessing**

Data preprocessing is a crucial step to clean, transform, and prepare the acquired data for machine learning. It involves handling missing values, removing outliers, standardizing or normalizing features, handling categorical variables, and addressing any inconsistencies or errors in the data. Preprocessing ensures the data is in a suitable format for analysis and model training.

**Data Labeling and Annotation**

For supervised learning tasks, where you have labeled data, you may need to label or annotate the data. Labeling involves assigning the correct target values to each data instance. This process can be done manually or by utilizing crowd-sourcing platforms or specialized labeling tools. Labeling is essential for training models to make accurate predictions based on labeled examples.

**Data Privacy and Security**

Ensure that you handle the acquired data with appropriate security measures to protect sensitive or private information. Follow data privacy regulations and anonymize or pseudonymize the data when necessary to maintain confidentiality.

## Data Type for Deep Learning

The deeplearning data can be classified as structred and unstructured data. The Structured data refers to data that is organized in a tabular format, where each column represents a specific attribute or feature, and each row represents an individual record. Structured data is typically found in databases, spreadsheets, or CSV files.

Unstructured data refers to data that does not have a predefined format or organization. It includes text, images, audio, video, social media posts, sensor data, and more.

**Numerical Data**
If the data consists of numerical features, they can be directly used as inputs to the NN. Ensure that the data is preprocessed, standardized, or normalized if necessary.

**Categorical Data**
Categorical data can be one-hot encoded, where each category becomes a binary feature. This allows NN to process categorical information effectively.

**Text Data**
Textual data requires additional preprocessing steps, such as tokenization and vectorization. Techniques like word embeddings or bag-of-words can be used to convert text into numerical representations suitable for NN/RNN.

**Image Data**
Images are typically represented as multi-dimensional arrays. For NN, images can be flattened into a 1D vector or kept as 2D/3D arrays, depending on the complexity of the task. Images are represented as multi-channel arrays (height, width, and depth). Ensure that the input images have consistent sizes. Preprocessing steps like resizing and normalization are often applied.

**Time Series Data**
Time series data, such as stock prices or sensor readings, requires proper sequencing. Sequential chunks of data need to be organized, considering factors like window size and step size.

It's important to note that these classifications are not exhaustive, and there can be variations and combinations depending on the specific problem and the deep learning architecture being used. Understanding the data and its characteristics is crucial for effectively applying deep learning models and achieving optimal performance.

**Audio and Video Data**
Audio and video data can be preprocessed using techniques like spectrogram analysis or time-frequency representation for audio, and frame extraction for video data.

## Train, Test, & Validation Sets**

Train set - The actual dataset that we use to train the model (weights and biases in the case of a Neural Network). The model sees and learns from this data.

Validation Set - it is a set of data different from training set that is used to validate our model during training. The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters.The validation set is used to evaluate a given model, but this is for frequent evaluation.

Test Set - The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.The Test dataset provides the gold standard used to evaluate the model. It is only used once a model is completely trained.

## Output Layer
In deep learning, the output layer refers to the final layer of a neural network that produces the desired output or predictions. The design of the output layer depends on the nature of the problem being solved, such as classification, regression, or sequence generation.

Let's look at some examples to understand the concept of the output layer and loss in deep learning:

## Classification
In a classification task, the goal is to assign input data points to specific classes or categories. The output layer typically uses a softmax activation function, which outputs a probability distribution over the classes. The number of neurons in the output layer corresponds to the number of classes.
For example, let's consider a neural network for classifying handwritten digits from 0 to 9. The output layer would have 10 neurons, each representing the probability of the input digit belonging to a particular class (e.g., neuron 1 represents the probability of the digit being 0, neuron 2 represents the probability of the digit being 1, and so on).

## Regression
In regression tasks, the objective is to predict a continuous numerical value. The output layer typically consists of a single neuron without any activation function. The output of this neuron directly represents the predicted value.
For instance, let's say we want to predict the price of a house based on its features like size, number of rooms, and location. In this case, the output layer would contain one neuron that provides the predicted price as the output.

## Sequence
In sequence generation tasks, such as language modeling or machine translation, the goal is to generate a sequence of output elements. The output layer is typically composed of multiple neurons, each representing the probability distribution of the possible output elements at each position in the sequence.
For example, in machine translation, the output layer may consist of multiple neurons, where each neuron represents a specific word or token in the target language. The network predicts the probabilities of different words at each position, allowing for the generation of the translated sequence.

## Loss function
The loss function measures the discrepancy between the predicted output and the actual target output. It quantifies the error of the model during training, allowing the model to adjust its parameters to minimize this error. The choice of the loss function depends on the problem type.

In classification tasks, a common loss function is the categorical cross-entropy, which measures the dissimilarity between the predicted probabilities and the true labels. In regression tasks, the mean squared error (MSE) loss function is often used to calculate the average squared difference between the predicted and actual values.

For example, if we use the categorical cross-entropy loss function for the digit classification task, the loss would penalize the network more if it assigns high probability to a wrong class and less if it assigns high probability to the correct class.

## Cross Entropy Loss
The cross-entropy loss is a commonly used loss function in classification tasks within deep learning. To understand its intuition, let's consider the context of binary classification, where we have two classes: positive and negative.

The goal of a binary classifier is to assign a probability to each class for a given input and predict the class with the highest probability. The cross-entropy loss helps to measure the dissimilarity between the predicted probabilities and the true labels.

Intuitively, the cross-entropy loss can be thought of as a way to quantify how surprised or uncertain the model is about the true class. If the predicted probabilities closely match the true probabilities (i.e., the true label is 1, and the model predicts a high probability for the positive class or vice versa), the cross-entropy loss will be low. On the other hand, if the predicted probabilities are far from the true probabilities, the loss will be high.

## Binary Cross Entropy Loss

Here's a more formal explanation of the cross-entropy loss calculation for binary classification:

Let's denote the true probability of the positive class as p and the predicted probability of the positive class as q. Then, the cross-entropy loss L is calculated as:

L = - (p * log(q) + (1 - p) * log(1 - q))

When the true label is 1, the loss equation becomes:

L = - log(q)

In this case, as q (the predicted probability of the positive class) approaches 1, the loss approaches 0. This means that the model is confident and correct in predicting the positive class. Conversely, as q approaches 0, the loss approaches infinity, indicating a high penalty for incorrectly predicting the negative class.

Similarly, when the true label is 0, the loss equation becomes:

L = - log(1 - q)

In this case, as q approaches 0, the loss approaches 0, indicating a correct prediction of the negative class. Conversely, as q approaches 1, the loss approaches infinity, penalizing incorrect predictions of the positive class.

The cross-entropy loss takes into account the probability distribution over the classes and penalizes the model more for predictions that diverge from the true labels. By minimizing this loss, the model learns to adjust its parameters to make more accurate predictions and converge towards the true underlying distribution of the data.

It's worth noting that the cross-entropy loss can be extended to multi-class classification problems using a similar principle, but with the summation of losses over all classes.

## Multiclass Cross Entropy Loss
Let's consider a multiclass classification problem with three classes: Class A, Class B, and Class C. The goal is to predict the correct class for a given input.

In multiclass classification, the cross-entropy loss extends the concept of binary cross-entropy to multiple classes. The intuition remains the same: to measure the dissimilarity between the predicted probabilities and the true labels. The loss encourages the model to assign higher probabilities to the correct class and lower probabilities to the incorrect classes.

To calculate the cross-entropy loss for multiclass classification, we use the following formula:

L = - (y1 * log(q1) + y2 * log(q2) + y3 * log(q3))

where yi represents the true probability (either 0 or 1) of class i and qi represents the predicted probability of class i.

Let's consider an example to illustrate this. Suppose we have an input image that belongs to Class B, and the true label is [0, 1, 0] (indicating that the image is not Class A, is Class B, and is not Class C).

Now, let's say the model predicts the following probabilities for the three classes: [0.1, 0.6, 0.3].

To calculate the cross-entropy loss, we substitute the values into the formula:

L = - (0 * log(0.1) + 1 * log(0.6) + 0 * log(0.3))
= - (0 + (-0.511) + 0)
≈ 0.511

In this case, the cross-entropy loss is approximately 0.511. A lower loss indicates that the predicted probabilities are closer to the true probabilities, and the model is performing well.

The model learns by minimizing this loss during training. By adjusting its parameters, it tries to reduce the difference between the predicted probabilities and the true probabilities, leading to more accurate predictions.

In summary, the cross-entropy loss for multiclass classification measures the dissimilarity between the predicted and true probability distributions. It helps guide the model to make better predictions by penalizing larger deviations and encouraging the model to assign higher probabilities to the correct class and lower probabilities to the incorrect classes.

## One-Hot Encodings
It transforms our categorical labels into vectors of zeroes and ones.The length of these vectors is equal to the number
of classes or categories that our model is expected to classify so if we categorize cat and dogs images then one hot encoded vectors that correspond to these classes will be each of length 2.

the cat vector=[1,0],cat corresponds to 1st element

the dog vector=[0,1],dog corresponds to 2nd element


for image labelled as cat the model will interpret cat as vector[1,0].

for image labelled as dog the model will interpret dog as vector[0,1].

## Epoch

In deep learning, we deal with huge data that may not be processed in one go. We handle the entire data by dividing the dataset in smaller group called batches. The batch size is the number of training examples processed in one iteration.

One iteration refers to a single update step of the model's parameters (weights and biases). This happens after the model processes one batch of training data and computes the gradient of the loss function with respect to the model's parameters, followed by an optimization step (e.g., using gradient descent).

This arrangement lead to introduce the term Epoch. An epoch refers to one complete pass through the entire training dataset by the model.

Let us understand this by a simple example:

Total training examples = 200

Let's assume a batch size of = 25

It takes 8 iterations to process all 200 training examples because each iteration processes 25 examples (one batch).
Each iteration updates the model parameters once. After 8 iterations, the model has seen all 200 examples, completing 1 epoch.

If you train for 40 iterations, 5 epoch will be completed and this means the training process has gone through the dataset 5 times.