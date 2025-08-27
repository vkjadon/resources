## Binary Classification using Logistic Rgression

Logistic Regression is similar to the Linear Regression except linear Regression is used to solve Regression problems, whereas Logistic regression is used to solve the classification problems. Logistic regression predicts the output of a categorical dependent variable. It can be either `Yes` or `No`, `0` or `1`, `true` or `False`, etc. but instead of giving the exact value as 0 and 1, it gives the `probabilistic values` which lie between 0 and 1.

1. Logistic Regression is a statistical model that uses Logistic Function to model a binary classification problem
2. `Logistic function` takes values either 0 or 1 when parameter is very large or small.
3. In Logistic Rgression, we intend to estimate the model parameters such that the probability of the output is true given the input features.

$\hat{y}=P(y=1|x)$

The probability is always between 0 and 1. So,

$0 \le \hat{y} \le 1$

## Setting up Problem

Let,  
$m$ : training examples  
$nx$ : Number of features  

![Logistic Regression Model](images/lr_model.png)

We will use $\mathbf{𝐱}^{(𝐢)}$  to denote the feature vector and  $\mathbf{𝐲}^{(𝐢)}$ to denote output variable.
Let us write $z$ for the output as linear combinations of weights and input features.

$z^{(i)} = w_1x^{(i)}_1+w_2x^{(i)}_2+.....+w_{nx}x^{(i)}_{nx}+b $   

Feature vector for $i^{th}$ training example:

$\mathbf{x}^{(i)} =\begin{pmatrix}{x}_1^{(i)} \\ {x}_2^{(i)} \\ \vdots \\ {x}_{nx}^{(i)}\end{pmatrix}$   

Output vector of $i^{th}$ training example:   

$y^{(i)}$ = (0 or 1)

$ \mathbf{y} = \begin{pmatrix} {y}^{(1)} & {y}^{(2)} & \cdots & {y}^{(m)}\end{pmatrix}$ 

Feature vector of the problem dataset:   

$ \mathbf{X} = \begin{pmatrix}\mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)}\end{pmatrix}$   

$ \mathbf{X} = \begin{pmatrix}{x}_1^{(1)} & {x}_1^{(2)} & \cdots & {x}_1^{(m)} \\ {x}_2^{(1)} & {x}_2^{(2)} & \cdots & {x}_2^{(m)} \\ \vdots & \vdots & \cdots & \vdots \\ {x}_{nx}^{(1)} & {x}_{nx}^{(2)} & \cdots & {x}_{nx}^{(m)} \end{pmatrix}$

Parameter vector :

$\mathbf{w} =\begin{pmatrix} {w}_1 \\ {w}_2 \\ \vdots \\ {w}_{nx} \end{pmatrix}, b $

## Generate dataset for Logistic Regression

Before going further deeper into the logistic regression problem, let us first talk about the dataset that you will be using in the session. You can use dataset of your own, but we will introduce `sklearn` library for generating synthetic data for a classification problem.

- Will use `make_classification` from `sklearn` classification dataset to generate random data.

```js
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
```

['make_classification' Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)

*Important Parameters:*

- n_samples: (int) The number of samples (default=100)
- n_features: (int) The total number of features ( default=20)
- n_classes: (int) The number of classes (or labels) of the classification problem (default=2)
- random_state: (int) Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls. (default=None)
- return_X_y: (bool) If True, a tuple (X, y) instead of a Bunch object is returned (default=True)

```js
X, y = make_classification(
    n_samples=50, 
    n_features=1, 
    n_informative=1,
    n_redundant=0, 
    n_classes=2, 
    n_clusters_per_class=1)
plt.scatter(X, y, c=y, cmap='bwr')
plt.show()
```

It is must to assign the values of `n_informative` and `n_redundant` to satisfy the condition:

> n_features $\ge$ n_informative + n_redundant + n_repeated

Another important condition that must be satisfied is:

> n_classes * n_clusters_per_class $\le$ 2**n_informative

![Dataset](images/dataset1D.png)

**Note** : your data may be different as the samples are randomly generated.

The size of X is (n_samples, n_features), so, in this case, it is (100, 20)

For our use the X matrix needs to be transposed to make it (20,100) i.e ($n_x$, $m$). The 100 is the default features and 20 is the default number of features.

```js
X, y = make_classification(
    n_samples=500, 
    n_features=2, 
    n_informative=2,
    n_redundant=0, 
    n_classes=3, 
    n_clusters_per_class=1,
    random_state=42)

plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
plt.title("n_classes=3")
plt.show()
```
In the figure below, the classes are three in different colors (Yellow, Green and Brown). In this case the plot should exactly match as we have used seed based random generator as specified by `random_state`. The seed is `42`.

![Dataset](images/dataset2DC3.png)


```js
X, y = make_classification(
    n_samples=500, 
    n_features=2, 
    n_informative=2,
    n_redundant=0, 
    n_classes=2, 
    n_clusters_per_class=2,
    random_state=42)

plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
plt.title("n_clusters_per_class=2")
plt.show()
```

In the figure below, there are 2 classes in two different colors (Red and Blue) and each classes are having two sub-classes. The plot should exactly match in this case also.

![Dataset](images/dataset2DC2SC2.png)

```js
data_set_x_orig, data_set_y_orig = make_classification(
  n_samples=1000,
  n_features=1, 
  n_informative=1, 
  n_redundant=0,
  n_classes=2,
  n_clusters_per_class=1,
  random_state=42)
#print(data_set_x_orig.shape)
#print(data_set_x_orig[:,0])
```

```js
plt.scatter(data_set_x_orig[:,0], data_set_y_orig, c=data_set_y_orig, cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression Data')
plt.show()
```

![Logistic Regression Model](images/lr_model.png)

```py
def initialize_with_zeros(features):

    w = np.zeros(features).reshape(features,1)
    b = 0.0

    return w, b
```
$z^{(i)} = \mathbf{w}^T \mathbf{x}^{(i)} + b$  

This can be implemented through the following function

```js
def forward_linear(x, w, b):
  z = np.dot(w.T, x) + b
  return z
```

In Logistic regression, instead of fitting a regression line, we fit an "S" shaped logistic function, which predicts values between 0 and 1.

$\hat{y}^{(i)} = a^{(i)} = σ (z^{(i)}) = \frac {1}{1+e^{-z^{(i)}}}$  

```py
def activation(z):
  return 1/(1 + np.exp(-z))
```

The squared error function for the logistic function may result in non-convex function. 


The gradient of the squared error involve products of σ(z), leading to potential non-convexity. Hence the other function is used as loss function for as below:

$$L(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})$$  

The above cost function is covex and hence we can have a global minima.

The sum of all the loss over entire training set is called the cost. The cost function is therefore computed by summing over all training examples:  

$$J(\mathbf{w},b) = \frac{1}{m} \sum_{i=1}^m L(a^{(i)}, y^{(i)})$$

$$J = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$$

The cost is to be minimized and the optimum parameters are to be evaluated. We can use gradient descent. The update rule is as below:  

$ \mathbf{w} = \mathbf{w} - \alpha \frac {\partial J}{\partial \mathbf{w}}$  

$ b = b - \alpha \frac {\partial J}{\partial b}$  

Where,  
    $ \alpha$ : Learning Rate (0.0001, 0.001, 0.01...)

The goal is to learn $w$ and $b$ by minimizing the cost function $J$

## Gradient Descent

To understand the basics of the gradient descent, let us consider one training example and multiple features. Then we can write the forward propogation equations by dropping the superscript $i$:

$z = w_1x_1+w_2x_2+.....+w_{nx}x_{nx}+b $  

$a = \frac {1}{1+e^{-z}}$  

$L(a, y) =  - y \log(a) - (1-y) \log(1-a)$  

To find the upated values of the parameters, we have to find the gradients $ \frac{\partial L(a, y)}{\partial w_j}$, $ \frac{\partial L(a, y)}{\partial b}$.

The loss function is a function of $a$ and $y$, so we have to use chain rule going backward from the last step to first step.   

Note that $a$ is a function of $z$ and $z$ is a function of $w$.

$\frac{\partial L(a, y)}{\partial w_j}= \frac{\partial L(a, y)}{\partial a} \frac{\partial a}{\partial z}  \frac{\partial z}{\partial w_j}$

Let us try to evaluate the each term separately.  

$\frac{\partial L(a, y)}{\partial a}=-\frac {y}{a}+\frac {(1-y)}{(1-a)}$

Using 
$\frac{\partial (\frac{u(x)}{v(x)})}{\partial y}=\frac {vu'-uv'}{v^2}$, we evaluate

$ \frac{\partial a}{\partial z}=\frac{\partial }{\partial z} (\frac {1}{1+e^{-z}}) = \frac {e^{-z}}{(1+e^{-z})^2}=\frac {1}{1+e^{-z}}\frac {1+e^{-z}-1}{1+e^{-z}} =a(\frac {1+e^{-z}}{1+e^{-z}} - \frac {1}{1+e^{-z}})$

$ \frac{\partial a}{\partial z}=a(1-a)$

$ \frac{\partial z}{\partial w_j}=x_j$

$\frac{\partial L(a, y)}{\partial w_j}= x_j(a-y)$

Similarly,

$\frac{\partial L(a, y)}{\partial b}= (a-y)$

Let us expand the expressions for $m$ training examples by taking the mean of the sum over all the training examples

$$ \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum \limits _{i=1} ^m (a^{(i)}-y^{(i)}){x}^{(i)}_j$$  

$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum \limits _{i=1} ^m (a^{(i)}-y^{(i)})$$

In matrix form,  
$$ \frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m}\mathbf {X(a-y)}^T$$

$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})$$

## Python implementation

<pre style="white-space:pre-line">
1. Arrange the input feature matrix $ \mathbf{X}.shape = (nx \times m)$    
2. Arrange the output vector $\mathbf{y}.shape=(1 \times m)$  
3. Assume learning rate and number of iterations
4. Initialize the weight vector $\mathbf{w}.shape=(nx \times 1)$ We can initialize these with `zeros` for linear regression.
5. Initialize the bias as zero (scalar)
6. Loop over iteration  
(a) Calculate predicted value array for assumed/updated parameters  
$ \mathbf{z}=\mathbf{w}^T\mathbf{X}+b$  
(b) Calculate sigmoid  
$\mathbf a=Sigmoid(\mathbf z)$   
(c) Calculate the partial derivative of Cost Function with respect to weights  
$ \frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}(\mathbf{a-y})^T$  
(d) Calculate the partial derivative of Cost Function with respect to bias. This is evaluated by summing the difference of predicted value and actual value and taking mean.  
$ \frac{\partial J}{\partial b} = \frac{1}{m} Sum(\mathbf{a-y})$  
(e) Update the weight and bias  
$ \mathbf{w} = \mathbf{w} - \alpha \frac {\partial J}{\partial \mathbf{w}}$  
$ b = b - \alpha \frac {\partial J}{\partial b}$

</pre>

## Logistic Regression Algorithm



### Split the dataset into training and test dataset

```js
x_train, x_test, y_train, y_test = train_test_split(data_set_x_orig, data_set_y_orig, train_size=0.8, test_size=0.2, random_state=1)
```
### Preparing Matrices

```js
m_train=x_train.shape[0]
x_train=x_train.T
y_train=y_train.reshape(y_train.shape[0],1).T
print(y_train.shape)
m_test=x_test.shape[0]
x_test=x_test.T
y_test=y_test.reshape(y_test.shape[0],1).T
```
### Training Code

```js
learning_rate=0.005
max_iteration=40000

w=np.zeros(x_train.shape[0]).reshape(x_train.shape[0],1)
cost=np.zeros((max_iteration))
b=0.0
for i in range(max_iteration):
  z=np.dot(w.T, x_train)
  A=1/(1+np.exp(-z))
  #print(A, A.shape)
  cost[i]=-np.sum((y_train*np.log(A)+(1-y_train)*np.log(1-A)),axis=1)/m_train
  #print(" Cost", cost)
  dw=np.dot(x_train,(A-y_train).T)/m_train
  db=(np.sum((A-y_train), dtype=np.float64))/m_train
  #print("Gradients dw, b", dw,db)
  w=w-learning_rate*dw
  b=b-learning_rate*db
print(w, b)
```
### Learning Curve

```js
# Plot learning curve (with costs)
plt.plot(cost)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate")
plt.show()
```

### Prediction Code

```js
#Prediction
A=1/(1+np.exp(-np.dot(w.T, x_train)))
Y_prediction_train=np.zeros((1,x_train.shape[1]))
for i in range(A.shape[1]):
  if A[0,i] > 0.5:
    Y_prediction_train[0,i]=1
  else:
    Y_prediction_train[0,i]=0
print("Training accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_train-y_train))*100))
A=1/(1+np.exp(-np.dot(w.T, x_test)))
Y_prediction_test=np.zeros((1,x_test.shape[1]))
for i in range(A.shape[1]):
  if A[0,i] > 0.5:
    Y_prediction_test[0,i]=1
  else:
    Y_prediction_test[0,i]=0
print("Testing accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_test-y_test))*100))
# print(y_train)
```
### Confusion Matrix

```js
# Show the Confusion Matrix
#print(y_test, Y_prediction_test)
confusion_matrix(y_test.T, Y_prediction_test.T)
```