## Cat Classifier using Shallow Neural Network

The network below is called as 2-layer Neural Network (NN). The input layer is not counted by convention to nomenclate the network. The layers between output and input layers are called as hidden layer. In the NN shown below is having one hidden layer and the output layer.

![Neural Network](images/nn2l.png)

We have developed the basic understanding of the processing happening in the neuron or hidden unit. Each unit is supposed to perform two tasks. First, it represnt the input featues and the weights as linear combination and this computation is called as the linear part of hidden unit computaion. The other part of the computation is the activation using non-linear activation function. This can be represneted in general by the following:  

## Input Dataset as per Matrix Notation

Input Feature vector for $i^{th}$ training example:

$\mathbf{x}^{(i)} = \mathbf{a}^{[0](i)} = \begin{pmatrix} {x}_1^{(i)} \\ {x}_2^{(i)} \\ \vdots \\ {x}_{nx}^{(i)} \end{pmatrix}= \begin{pmatrix} {a}_1^{[0](i)} \\ {a}_2^{[0](i)} \\ \vdots \\ {a}_{nx}^{[0](i)} \end{pmatrix} $   

Input Feature vector of the problem dataset:   

$ \mathbf{X} = \mathbf{A}^{[0]}= \begin{pmatrix} \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(i)}
\end{pmatrix}$   

$ \mathbf{X} = \mathbf{A}^{[0]} = \begin{pmatrix} {x}_1^{(1)} & {x}_1^{(2)} & \cdots & {x}_1^{(m)} \\ {x}_2^{(1)} & {x}_2^{(2)} & \cdots & {x}_2^{(m)} \\ \vdots & \vdots & \cdots & \vdots \\ {x}_{nx}^{(1)} & {x}_{nx}^{(1)} & \cdots & {x}_{nx}^{(m)} \end{pmatrix}, \mathbf{X} \in \mathbf R ^{nx \times m}$  

**Let the input features be 3 to develop the intuition about the forward and backward propogation.**

```py
! git clone https://github.com/vkjadon/utils/
```
```js
import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.public_tests import *
```
## Fetch Dataset from Kaggle
- Import data from Kaggle
- Use <a href="https://www.kaggle.com/muhammeddalkran/catvnoncat" target="_blank"> this Link </a>  

From the download button, copy the following code and execute to download the data.

```js
import kagglehub

# Download latest version
path = kagglehub.dataset_download("muhammeddalkran/catvnoncat")

print("Path to dataset files:", path)
```

## Load Dataset

```js
train_dataset = h5py.File( path + '/catvnoncat/train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"]) # your train set features
train_set_y_orig = np.array(train_dataset["train_set_y"]) # your train set labels

num_px = train_set_x_orig.shape[1]
num_py = train_set_x_orig.shape[2]
nx = num_px * num_py * 3
m_train = train_set_x_orig.shape[0]

train_set_x=train_set_x_orig.reshape(m_train, -1).T
train_set_y = train_set_y_orig.reshape((1, m_train))
# train_set_x=train_set_x_orig.reshape(-1, m_train)

X_train = train_set_x / 255.
y_train=train_set_y

nx, m_train = X.shape
nh = 4
ny = y.shape[0]

```

## Forward Linear Computation : Layer - 1

**Node-1 :** $[z^{[1](i)}_1]$  

![Neural Network](images/neuron_process.png)

$ z^{[1](i)}_1 = w_{11}^{[1]}x^{(i)}_1+w_{12}^{[1]}x^{(i)}_2+w_{13}^{[1]}x^{(i)}_3 + b^{[1]}_1 $   

**Node-2 :** $[z^{[1](i)}_2]$  

$ z^{[1](i)}_2 = w_{21}^{[1]}x^{(i)}_1+w_{22}^{[1]}x^{(i)}_2+w_{23}^{[1]}x^{(i)}_3 + b^{[1]}_2 $   

**Node-3 :** $[z^{[1](i)}_3]$  

$ z^{[1](i)}_3 = w_{31}^{[1]}x^{(i)}_1+w_{32}^{[1]}x^{(i)}_2+w_{33}^{[1]}x^{(i)}_3 + b^{[1]}_3 $   

**Node-4 :** $[z^{[1](i)}_4]$  

$ z^{[1](i)}_4 = w_{41}^{[1]}x^{(i)}_1+w_{42}^{[1]}x^{(i)}_2+w_{43}^{[1]}x^{(i)}_3 + b^{[1]}_4 $  


We can write the above equations in matrix form as below

$\begin{pmatrix} {z}_1 \\ {z}_2 \\ z_3 \\ {z}_{4}\end{pmatrix}^{[1](i)}= \begin{pmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \\ w_{41} & w_{42} & w_{43}\end{pmatrix}^{[1]} \begin{pmatrix} x_1 \\ x_2 \\ x_{3} \end{pmatrix}^{(i)} + \begin{pmatrix} b_1 \\ b_2 \\ b_3 \\ b_{4} \end{pmatrix}^{[1]}$

The superscripts $[1]$ and $(i)$ for the layer and the training example are taken out and placed as superscript on the parenthesis to represent that it is applicatble to all the elements of the array/matrix.

Let us define the vectors

$\mathbf{z}^{[1](i)} = \begin{pmatrix} z_1 \\ z_2 \\ z_3 \\ z_4 \end{pmatrix}^{[1] (i)}$

$\mathbf{W}^{[1]} = \begin{pmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \\ w_{41} & w_{42} & w_{43}\end{pmatrix}^{[1]}$

$\mathbf{x}^{(i)} = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}^{(i)}$

and

$\mathbf{b}^{[1]} = \begin{pmatrix} b_1 \\ b_2 \\ b_3 \\ b_4 \end{pmatrix}^{[1]}$

We can consider input features as activation of zeroth layer.

$\mathbf{x}^{(i)} = \begin{pmatrix} a_1 \\ a_2 \\ a_3 \end{pmatrix}^{[0](i)}$

$\mathbf W^{[1]}.\text shape()=(n_L, n_{L-1})$
  

```js
def initialize_parameters(nx, nL, ny):
  random_state = 2
  rng = np.random.default_rng(random_state)
  W1 = rng.standard_normal((nL, nx)) * 0.01
  b1 = np.zeros((nL, 1))
  W2 = rng.standard_normal((ny, nL)) * 0.01
  b2 = np.zeros((ny, 1))
  parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2}
  return parameters
```
The vectorized equation of the forward linear coputation can be written as:  

$\mathbf{z}^{[1] (i)} = \mathbf{W}^{[1]}\mathbf{x}^{(i)} + \mathbf{b}^{[1]}$ 

This can be implemented using `numpy dot()` as under using common variables:   

```js
def forward_linear(A, W, b):
  Z = W.dot(A) + b
  assert(Z.shape == (W.shape[0], A.shape[1]))
  cache = (A, W, b)
  return Z, cache
```
   
Alternatively, we can also deduce the above expression for the linear output of hidden layer as below:

write the linear output equation $ z^{[1](i)}_1 = w_{11}^{[1]}x^{(i)}_1+w_{12}^{[1]}x^{(i)}_2+w_{13}^{[1]}x^{(i)}_3 + b^{[1]}_1 $ as below: 

$z^{[1](i)}_1\ =\begin{pmatrix} w_{11} & w_{12} & w_{13} \end{pmatrix}^{[1]} \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}^{(i)} + b^{[1]}_1$  

As per convention we have to arrange weights in a column. So, the weight vector of first neuron of hidden layer is

$ \mathbf{w}^{[1]}_1 = \begin{pmatrix} w_{11} \\ w_{12} \\ w_{13} \end{pmatrix}^{[1]} $

In matrix form   

$z^{[1](i)}_1 = \mathbf{w}_1^{[1]T}\mathbf{x}^{(i)} + b^{[1]}_1$

Replacing input feature with the activation of layer $[0]$

$z^{[1](i)}_1 = \mathbf w_1^{[1]T}\mathbf a^{[0](i)} + b^{[1]}_1$

Similarly for other neurons of the hidden layer

$ z^{[1](i)}_2 = \mathbf{w}_2^{[1]T}\mathbf{a}^{[0](i)} + b^{[1]}_2 $   

$ z^{[1](i)}_3 = \mathbf{w}_3^{[1]T}\mathbf{a}^{[0](i)} + b^{[1]}_3 $  

$ z^{[1](i)}_4 = \mathbf{w}_4^{[1]T}\mathbf{a}^{[0](i)} + b^{[1]}_4 $  

We can write the above set of equations in matrix form as under:  

$ \mathbf z^{[1] (i)} = \begin{pmatrix} z_1 \\ z_2 \\ z_3 \\ z_4 \end{pmatrix}^{[1](i)} = \begin{pmatrix} \mathbf w_1  \\ \mathbf w_2 \\ \mathbf w_3 \\ \mathbf w_4 \end{pmatrix}^{[1]T} \mathbf a^{[0](i)}+ \begin{pmatrix} b_1 \\ b_2 \\ b_3 \\ b_4 \end{pmatrix}^{[1]} $ 

The vectorized form is written as below    

  $\mathbf{z}^{[1](i)} = \mathbf{W}^{[1]}\mathbf{a}^{[0](i)} + \mathbf{b}^{[1]}$

This is the same as that of expression derived earlier.

$\mathbf{W}^{[1]} = \begin{pmatrix} \mathbf w_1^T \\ \mathbf w_2^T \\ \mathbf w_3^T \\ \mathbf w_4^T\end{pmatrix}^{[1]} $


## Forward Activated Computation : Layer - 1

We use $sigmoid()$ activation function when we need to calculate probability and the output is converted into 0 and 1 with the use of some threshold value.   

We use $tanh()$ function for the hidden layer as it maps negative values with negative and zero with zero. The output of the $tanh()$ function ranges between $-1$ to $+1$. The shape of $tanh()$ is also s-shaped as the shape of $sigmoid()$ activation is but it is shfted to map 0 with 0. This activation maps with the normalized input data and speed up the training convergence.

The $tanh(z)$ is given by $tanh(z)=\large \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$.  

The activated output of each node of Layer - 1 is obtained by implementing `tanh()` function on the linear output of the respective nodes:

$\mathbf{a}^{[1](i)} = tanh(\mathbf{z}^{[1](i)})$.   


$\large \frac{\partial[tanh(z)]}{\partial z}=\small 1-[tanh(z)]^2$.  


## Forward Linear Computation : Layer - 2

The layer - 2 in this case is the output layer, so the treatment given here for this layer will be used for output layer in deep networks

**Node-1** : $[z^{[2](i)}_1]$  

$ z^{[2] (i)} = w_{11}^{[2]}a^{[1] (i)}_1+w_{12}^{[2]}a^{[1] (i)}_2+w_{13}^{[2]}a^{[1] (i)}_3 + w_{14}^{[2]}a^{[1] (i)}_4 + b^{[2]}_1 $   

$ z^{[2] (i)} = \mathbf{w}^{[2]T}\mathbf{a}^{[1] (i)} + b^{[2]} $  

Note that $\mathbf w^{[2]}.shape=(4,1)$ but $W.shape=(n^{[l]}, n^{[l-1]})$. So, $W^{[2]}.shape=(1,4)$, this gives $W^{[2]}=\mathbf w^{[2]T}$  

$\mathbf{z}^{[2] (i)} = \mathbf{W}^{[2]} \mathbf{a}^{[1] (i)} + \mathbf{b}^{[2]}$.   



## Forward Activated Computation : Layer - 2

$a^{[2] (i)}=\sigma (z^{[2] (i)}) $

We will use $sigmoid()$ activation function for the output layer as we need the binary classification on the basis of the probability of output being true for give data.  

$\mathbf{a}^{[2] (i)} = sigmoid(\mathbf{z}^{[2] (i)})$.

## Equations for Implementing Forward Steps

Let us re-write the set of four vectorized equations developed above for hidden layer and output layer for the $(i)^{th}$ training example:  

**Hidden Layer : Layer - 1**  

$\mathbf{z}^{[1](i)} = \mathbf{W}^{[1]} \mathbf{a}^{[0](i)} + \mathbf{b}^{[1]}$   

$\mathbf{a}^{[1](i)} = tanh(\mathbf{z}^{[1](i)})$   

**Output Layer : Layer - 2**  

$\mathbf{z}^{[2](i)} = \mathbf{W}^{[2]} \mathbf{a}^{[1](i)} + \mathbf{b}^{[2]}$   

$\mathbf{a}^{[2](i)} = sigmoid(\mathbf{z}^{[2](i)})$   

These four vectorized equations, two for each layer are to be used for python implementation.

## Trainable Parameters

The output of the forward step is the prediction for the assumed/initialized/updated parameters. Our aim is to push the predictions as close as possible to the truth labels. For that, you need to optimize the all possible traininable parameters of the model.

These parameters need to be initialized to start the forward propagation. The parameters in the example of 2-Layer with 3 input features and 4 nodes in the hidden layer for the binary classification problem are $\mathbf W^{[1]}, \mathbf W^{[2]}, \mathbf b^{[1]}, b^{[2]}$.   

$ \mathbf W^{[1]}.shape=(4, 3)$  

$ \mathbf W^{[2]}.shape=(1, 4)$  

$ \mathbf b^{[1]}.shape=(4, 1)$  

$ \mathbf b^{[2]}.shape=(1, 1)$  

So, in all we have to optimize the 21 parameters for this network using the training data. We need to identify a loss function to know how accurate our predictions are from each iteration and that function should be able to implement the gradient descent algorithm efficiently.

## Loss Function   

As discusssed in the previous session of logistic regression, we choose binary cross entropy loss function. The function is defined as:   

$L(a, y) =  - y  \log(a^{[2]}) - (1-y)  \log(1-a^{[2]})$   

The loss is to be minimized using garient descent optimization method. In this, we evaluate the gradient of the function and update the parameter till we reach the global minima. Following general update rules are applied:   

$ \mathbf w = \mathbf w - \alpha \frac {\partial L}{\partial \mathbf w}$  

$ \mathbf b = \mathbf b - \alpha \frac {\partial L}{\partial \mathbf b}$  

Where,  
        $ \alpha$ : Learning Rate (0.0001, 0.001, 0.01...)

## Backward Activated Computation : Layer - 2

The parameters of output layer are $w_{11}^{[2]}, w_{12}^{[2]}, w_{13}^{[2]}, w_{14}^{[2]}, b^{[2]}$. The gradients are computed in the same manner as derived in case of logistic regression as the **sigmoid function** is the activation function on output layer.     

$\large \frac{\partial L}{\partial w_{11}^{[2]}}= \frac{\partial L}{\partial a^{[2]}} \frac{\partial a^{[2]}}{\partial z^{[2]}}  \frac{\partial z^{[2]}}{\partial w_{11}^{[2]}}=\frac{\partial L}{\partial z^{[2]}}\small a^{[1]}_1=(a^{[2]}-y)a^{[1]}_1$

$\large \frac{\partial L}{\partial w_{12}^{[2]}}= \frac{\partial L}{\partial a^{[2]}} \frac{\partial a^{[2]}}{\partial z^{[2]}}  \frac{\partial z^{[2]}}{\partial w_{12}^{[2]}}=\frac{\partial L}{\partial z^{[2]}}\small a^{[1]}_2=(a^{[2]}-y)a^{[1]}_2$

$\large \frac{\partial L}{\partial w_{13}^{[2]}}= \frac{\partial L}{\partial a^{[2]}} \frac{\partial a^{[2]}}{\partial z^{[2]}}  \frac{\partial z^{[2]}}{\partial w_{13}^{[2]}}=\frac{\partial L}{\partial z^{[2]}}\small a^{[1]}_3=(a^{[2]}-y)a^{[1]}_3$

$\large \frac{\partial L}{\partial w_{14}^{[2]}}= \frac{\partial L}{\partial a^{[2]}} \frac{\partial a^{[2]}}{\partial z^{[2]}}  \frac{\partial z^{[2]}}{\partial w_{14}^{[2]}}=\frac{\partial L}{\partial z^{[2]}}\small a^{[1]}_4=(a^{[2]}-y)a^{[1]}_4$

$\large \frac{\partial L}{\partial b^{[2]}}= \frac{\partial L}{\partial a^{[2]}} \frac{\partial a^{[2]}}{\partial z^{[2]}}  \frac{\partial z^{[2]}}{\partial b^{[2]}}=\frac{\partial L}{\partial z^{[2]}}\small=(a^{[2]}-y)$

***Vectorized Gradient Equations for Output Layer***

The first four equations of gradient descent gives us the gradient of the loss with respect to the weights of output layer. These equations can be written in matrix form as below:


$\large \frac{\partial L}{\partial \mathbf W^{[2]}} = \begin{pmatrix} \frac{\partial L}{\partial w_{11}^{[2]}} & \frac{\partial L}{\partial w_{12}^{[2]}} & \frac{\partial L}{\partial w_{13}^{[2]}} & \frac{\partial L}{\partial w_{14}^{[2]}} \end{pmatrix}$

$\large \frac{\partial L}{\partial \mathbf W^{[2]}} =  \begin{bmatrix} (a^{[2]}-y)a^{[1]}_1 & (a^{[2]}-y)a^{[1]}_2 & (a^{[2]}-y)a^{[1]}_3 & (a^{[2]}-y)a^{[1]}_4 \end{bmatrix}$

The shape of the gradient is the same as that of the weight matrix which is $(n^{[2]}, n^{[1]})$ or $(1,4)$

$\large \frac{\partial L}{\partial \mathbf W^{[2]}}\small=(a^{[2]}-y)\begin{pmatrix} a^{[1]}_1 & a^{[1]}_2 & a^{[1]}_3 & a^{[1]}_4 \end{pmatrix}$   

The above matrix form can be written in vector form as:   

$\large \frac{\partial L}{\partial \mathbf W^{[2]}}=\small (a^{[2]}-y)\mathbf a^{[1]T}$   

$ \mathbf W^{[2]} = \mathbf W^{[2]} - \alpha \frac {\partial L}{\partial \mathbf W^{[2]}}$   

$ b = b - \alpha \frac {\partial L}{\partial b}$   

The above three vectorized equations are to be used for python implementation.

## Calculating Gradients for Hidden Layer

The parameters of hidden layer are 12 elements of $\mathbf W^{[1]}$ matrix and 4 elements of $\mathbf b^{[1]}$ matrix. It is obvious that the shape $\frac {\partial L}{\partial \mathbf W^{[1]}}$ will be the same of $\mathbf W^{[1]}$. Similarly, the shape $\frac {\partial L}{\partial \mathbf b^{[1]}}$ will be the same of $\mathbf b^{[1]}$.

So, the shape of the $\frac {\partial L}{\partial \mathbf W^{[1]}}$ is $(n^{[1]}, n^{[0]})$ or $(4,3)$

Let us write the gradient matrix:   

$\large \frac {\partial L}{\partial \mathbf W^{[1]}} = \begin{pmatrix} \frac {\partial L}{\partial w_{11}^{[1]}} & \frac {\partial L}{\partial w_{12}^{[1]}} & \frac {\partial L}{\partial w_{13}^{[1]}} \\ \frac {\partial L}{\partial w_{21}^{[1]}} & \frac {\partial L}{\partial w_{22}^{[1]}} & \frac {\partial L}{\partial w_{23}^{[1]}} \\ \frac {\partial L}{\partial w_{31}^{[1]}} & \frac {\partial L}{\partial w_{32}^{[1]}} & \frac {\partial L}{\partial w_{33}^{[1]}} \\ \frac {\partial L}{\partial w_{41}^{[1]}} & \frac {\partial L}{\partial w_{42}^{[1]}} & \frac {\partial L}{\partial w_{43}^{[1]}} \end{pmatrix};$

$\large \frac {\partial L}{\partial \mathbf b^{[1]}} = \begin{pmatrix} \frac {\partial L}{\partial b_{1}^{[1]}} \\ \frac {\partial L}{\partial b_{2}^{[1]}} \\ \frac {\partial L}{\partial b_{3}^{[1]}} \\ \frac {\partial L}{\partial b_{4}^{[1]}} \end{pmatrix}$

To develop the expressions for the each of the elements of above two matrices, let us consider first element $\frac {\partial L}{\partial w_{11}^{[1]}}$ to build the concept.   

$\large \frac {\partial L}{\partial w_{11}^{[1]}}=(\frac{\partial L}{\partial a^{[2]}})(\frac{\partial a^{[2]}}{\partial z^{[2]}})(\frac{\partial z^{[2]}}{\partial a^{[1]}_{1}}) (\frac{\partial a^{[1]}_{1}}{\partial z^{[1]}_1}) (\frac{\partial z^{[1]}_1}{\partial w^{[1]}_{11}})$

Let us evaluate these term one by one. Out of these, we have already computed the first two terms in the gradient calculation of the output layer. So, let us club these two and compute the other as under:

$\large \frac{\partial L}{\partial z^{[2]}}= \frac{\partial L}{\partial a^{[2]}} \frac{\partial a^{[2]}}{\partial z^{[2]}}=\frac{\partial L}{\partial z^{[2]}}\small=(a^{[2]}-y)$

$\large \frac{\partial z^{[2]}}{\partial a^{[1]}_{1}}\small=w^{[2]}_{11}$ : using $ z^{[2]} = w_{11}^{[2]}a^{[1]}_1+w_{12}^{[2]}a^{[1]}_2+w_{13}^{[2]}a^{[1]}_3 + w_{14}^{[2]}a^{[1]}_4 + b^{[2]}_1 $  

$\large \frac{\partial a^{[1]}_{1}}{\partial z^{[1]}_1}\small=g'(z^{[1]}_1)$

$\large \frac{\partial z^{[1]}_1}{\partial w^{[1]}_{11}}\small=x_1$ : using $ z^{[1]}_1 = w_{11}^{[1]}x_1+w_{12}^{[1]}x_2+w_{13}^{[1]}x_3 + b^{[1]}_1 $

Combining all the expression to get the desired gradient for updating the parameters associated with the first node of hidden layer:   

$\large \frac {\partial L}{\partial w_{11}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{11}g'(z^{[1]}_1)x_1$
  
Similarly, we can write the other elements:  

$\large \frac {\partial L}{\partial w_{12}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{11}g'(z^{[1]}_1)x_2$  

$\large \frac {\partial L}{\partial w_{13}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{11}g'(z^{[1]}_1)x_3$

The gradient to update the parameters associated with the second node of the hidden layer:  

$\large \frac {\partial L}{\partial w_{21}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{12}g'(z^{[1]}_2)x_1$

$\large \frac {\partial L}{\partial w_{22}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{12}g'(z^{[1]}_2)x_2$  

$\large \frac {\partial L}{\partial w_{23}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{12}g'(z^{[1]}_2)x_3$

The gradient to update the parameters associated with the third node of the hidden layer:  

$\large \frac {\partial L}{\partial w_{31}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{13}g'(z^{[1]}_3)x_1$

$\large \frac {\partial L}{\partial w_{32}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{13}g'(z^{[1]}_3)x_2$  

$\large \frac {\partial L}{\partial w_{33}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{13}g'(z^{[1]}_3)x_3$

The gradient to update the parameters associated with the fourth node of the hidden layer:  

$\large \frac {\partial L}{\partial w_{41}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{14}g'(z^{[1]}_4)x_1$

$\large \frac {\partial L}{\partial w_{42}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{14}g'(z^{[1]}_4)x_2$  

$\large \frac {\partial L}{\partial w_{43}^{[1]}}\small=(a^{[2]}-y)w^{[2]}_{14}g'(z^{[1]}_4)x_3$

Substituting the above calculated expression for each of the element into the matrix $\large \frac {\partial L}{\partial \mathbf W^{[1]}}$.

$\large \frac {\partial L}{\partial \mathbf W^{[1]}} \small= (a^{[2]}-y) \begin{pmatrix}w^{[2]}_{11}g'(z^{[1]}_1) x_1 & w^{[2]}_{11}g'(z^{[1]}_1) x_2 & w^{[2]}_{11}g'(z^{[1]}_1) x_3 \\ w^{[2]}_{12}g'(z^{[1]}_2) x_1 & w^{[2]}_{12}g'(z^{[1]}_2) x_2 & w^{[2]}_{12}g'(z^{[1]}_2) x_3 \\ w^{[2]}_{13}g'(z^{[1]}_3) x_1 & w^{[2]}_{13}g'(z^{[1]}_3) x_2 & w^{[2]}_{13}g'(z^{[1]}_3) x_3 \\ w^{[2]}_{14}g'(z^{[1]}_4) x_1 & w^{[2]}_{14}g'(z^{[1]}_4) x_2 & w^{[2]}_{14}g'(z^{[1]}_4) x_3 \end{pmatrix}$

$\large \frac {\partial L}{\partial \mathbf W^{[1]}} \small= (a^{[2]}-y) \begin{pmatrix} w^{[2]}_{11}g'(z^{[1]}_1) \\ w^{[2]}_{12}g'(z^{[1]}_2) \\ w^{[2]}_{13}g'(z^{[1]}_3) \\ w^{[2]}_{14}g'(z^{[1]}_4) \end{pmatrix} \begin{pmatrix} x_1 &  x_2 & x_3 \end{pmatrix}$.  

The above matrix equation can be written in vectorized form as below:    

$\large \frac {\partial L}{\partial \mathbf W^{[1]}} \small=(a^{[2]}-y)([\mathbf W^{[2]T}*g'(\mathbf z^{[1]})] \times \mathbf x^T)$.   

$\mathbf W^{[2]}$.shape = (1,4) from W.shape=$(n^{[2]}, n^{[1]})$.  

$g'(\mathbf z^{[1]})$.shape = (4,1).   

$\mathbf x^T$.shape=(1,3)

Please note that $[\mathbf W^{[2]T}*g'(\mathbf z^{[1]})]$ is the element wise multiplication (represnted by the *) and the $\times$ represnts the vector multiplication.   

Now we can use the update formula for gradient descent to optimize the cost.

## Back Propagation Equations  

**Output Layer**

$\large \frac{\partial L}{\partial \mathbf W^{[2]}}=\small (a^{[2]}-y)\mathbf a^{[1]T}$

$\large \frac{\partial L}{\partial \mathbf b^{[2]}}=\small (a^{[2]}-y)$

$ \mathbf W^{[2]} = \mathbf W^{[2]} - \alpha \frac {\partial L}{\partial \mathbf W^{[2]}}$   

$b^{[2]} = b^{[2]} - \alpha \frac {\partial L}{\partial b^{[2]}}$   

**Hidden Layer**

$\large \frac {\partial L}{\partial \mathbf W^{[1]}} \small=(a^{[2]}-y)([\mathbf W^{[2]T}*g'(\mathbf z^{[1]})] \times \mathbf x^T)$.   

$\large \frac {\partial L}{\partial \mathbf b^{[1]}} \small=(a^{[2]}-y)([\mathbf W^{[2]T}*g'(\mathbf z^{[1]})])$.   

$ \mathbf W^{[1]} = \mathbf W^{[1]} - \alpha \frac {\partial L}{\partial \mathbf W^{[1]}}$   

$b^{[1]} = b^{[1]} - \alpha \frac {\partial L}{\partial b^{[1]}}$   

The above vectorized equations are used for python implementation.



## Forward Propagation with m training examples   

When we consider $m$ training examples, we have to stack the data horizontally for each training examples. The size of the weight and bias matrix/vector are not affected by training examples. The $\mathbf {z, a}$ and $\mathbf x $ would be replaced with $\mathbf {Z, A}$ and $\mathbf X $ by stacking training examples horizontally. It may be noted that input feature matrix $\mathbf X$ can be written as $\mathbf A^{[0]}$.    

The concept of stacking training examples horizontally is used below to convert the equations developed for forward and backward propagation steps earlier, into the equations to handle $m$ training examples.

## Hidden Layer (Layer-1)

$\mathbf {Z}^{[1]} = \begin{pmatrix} z_1^{[1](1)} & z^{[1](2)}_1 & \cdots & z^{[1](m)}_1\\ z^{[1](1)}_2 & z^{[1](2)}_2 & \cdots & z^{[1](m)}_2\\ z^{[1](1)}_3 & z^{[1](2)}_3 & \cdots & z^{[1](m)}_3 \\ z^{[1](1)}_4 & z^{[1](2)}_4 & \cdots & z^{[1](m)}_4 \end{pmatrix}$

$ \mathbf{Z}^{[1]} = \begin{pmatrix} \mathbf{z}^{[1][1]} & \mathbf{z}^{[1](2)} & \cdots & \mathbf{z}^{[1](m)} \end{pmatrix}$  


$ \mathbf{Z}^{[1]} = \begin{pmatrix}
\mathbf{W}^{[1]} \mathbf{x}^{(1)} + \mathbf{b}^{[1]} & \mathbf{W}^{[1]} \mathbf{x}^{(2)} + \mathbf{b}^{[1]} & \cdots & \mathbf{W}^{[1]} \mathbf{x}^{(m)} + \mathbf{b}^{[1]}
\end{pmatrix}$

**Layer-1:Vectorized Linear Part**   

$ \mathbf{Z}^{[1]} = \mathbf{W}^{[1]} \mathbf X + \mathbf{b}^{[1]}$  

$ \mathbf{Z}^{[1]} = \mathbf{W}^{[1]} \mathbf{A^{[0]}} + \mathbf{b}^{[1]}$

**Layer-1:Vectorized Activation Part**   

$ \mathbf{A}^{[1]}=g (\mathbf{Z}^{[1]}) $    

We donot use signmoid as activation in the hidden layer. Rather we use $tan^{-1}(z)$, ReLU, leaky ReLU as activation function in the hidden layers. Sigmoid is the default choice for the output layer of the binary classification problem.

## Output Layer (Layer-2)

$ \mathbf{Z}^{[2]} = \begin{pmatrix} z^{[2][1]} & z^{[2](2)} & \cdots & z^{[2](m)} \end{pmatrix}$  

**Layer-2 Vectorized Linear Part**   

$ \mathbf{Z}^{[2]} = \mathbf{W}^{[2]} \mathbf{A}^{[1]} + \mathbf{b}^{[2]}$

**Layer-2 Activation Part**

$ \mathbf{A}^{[2]}=\sigma (\mathbf{Z}^{[2]}) $

**Forward Propagation Equations for implementing the logistic regression model**  

$\mathbf Z^{[1]} =  \mathbf W^{[1]} \mathbf X + \mathbf b^{[1]}$  

$\mathbf A^{[1]} = \tanh(\mathbf Z^{[1]})$  

$\mathbf Z^{[2]} = \mathbf W^{[2]} \mathbf A^{[1]} + b^{[2]}$  

$ \mathbf A^{[2]} = Sigmoid(\mathbf Z^{[2]})$

The general shapes of various vectors and matrices are  

$(\mathbf X=\mathbf A^{[0]}).shape=(n^{[0]},m);$

$ \mathbf W^{[1]}.shape=(n^{[1]}, n^{[0]}); \mathbf W^{[2]}.shape=(n^{[2]}, n^{[1]})$

$ \mathbf b^{[1]}.shape=(n^{[1]}, 1); \mathbf b^{[2]}.shape=(n^{[2]}, 1)$

$ \mathbf Z^{[1]}.shape=(n^{[1]},m); \mathbf Z^{[2]}.shape=(n^{[2]}, m)$  

$ \mathbf A^{[1]}.shape=(n^{[1]}, m); \mathbf A^{[2]}.shape=(n^{[2]}, m)$  


For the given network, we have,   

$n^{[0]}=3; n^{[1]}=4; n^{[2]}=1$

Therefore, we can find the shape of the vectors and matrices as:    

$ \mathbf A^{[0]}=(3,m); \mathbf A^{[1]}=(4, m); \mathbf A^{[2]}=(1, m)$  

$ \mathbf Z^{[1]}=(4,m); \mathbf Z^{[2]}=(1, m)$  

$ \mathbf W^{[1]}=(4, 3); \mathbf W^{[2]}=(1, 3) \mathbf b^{[1]}=(4, 1); \mathbf b^{[2]}=(1, 1)$

## Cost Function

The $\mathbf A^{[2]}$ contains $\mathbf a^{[2](i)}$ for all examples. We can now compute the cost function as follows:

$$ \mathbf J = - \frac{1}{m} \sum\limits_{i = 1}^{m} \large [ \small y^{(i)}\log a^{[2] (i)} + (1-y^{(i)})\log (1- a^{[2] (i)}) \large ] $$

**Parameters to be optimized are $\mathbf W^{[1]}, \mathbf W^{[2]}, \mathbf b^{[1]}, \mathbf b^{[2]}$**.     

*We can use Gradient Descent for minimizing cost to get the optimized values of the parameters.*

**Note the shapes on the Parameters ($\mathbf W^{[1]}, \mathbf W^{[2]}, \mathbf b^{[1]}, \mathbf b^{[2]})$**.   

$ \mathbf W^{[1]}.shape=(n^{[1]}, n^{[0]})$  

$ \mathbf W^{[2]}.shape=(n^{[2]}, n^{[1]})$  

$ \mathbf b^{[1]}.shape=(n^{[1]}, 1)$  

$ \mathbf b^{[2]}.shape=(n^{[2]}, 1)$  
## Back Propagation to evaluate Gradients

We need to evaluate the gradients of the cost function with respect to the parameters.   

$\large \frac{\partial \mathbf {J} }{ \partial \mathbf W^{[2]} }, \frac{\partial \mathbf {J} }{ \partial b^{[2]}}, \frac{\partial \mathbf {J} }{ \partial \mathbf W^{[1]}}, \frac{\partial \mathbf {J} }{ \partial \mathbf b^{[1]} }$

## Back Propagation Equations  

**Output Layer**

$\large \frac{\partial L}{\partial \mathbf W^{[2]}}=\small (a^{[2]}-y)\mathbf a^{[1]T}$

$\large \frac{\partial L}{\partial \mathbf b^{[2]}}=\small (a^{[2]}-y)$

$ \mathbf W^{[2]} = \mathbf W^{[2]} - \alpha \frac {\partial L}{\partial \mathbf W^{[2]}}$   

$b^{[2]} = b^{[2]} - \alpha \frac {\partial L}{\partial b^{[2]}}$   

**Hidden Layer**

$\large \frac {\partial L}{\partial \mathbf W^{[1]}} \small=(a^{[2]}-y)([\mathbf W^{[2]T}*g'(\mathbf z^{[1]})] \times \mathbf x^T)$.   

$\large \frac {\partial L}{\partial \mathbf b^{[1]}} \small=(a^{[2]}-y)([\mathbf W^{[2]T}*g'(\mathbf z^{[1]})])$.   

$ \mathbf W^{[1]} = \mathbf W^{[1]} - \alpha \frac {\partial L}{\partial \mathbf W^{[1]}}$   

$b^{[1]} = b^{[1]} - \alpha \frac {\partial L}{\partial b^{[1]}}$   

The above vectorized equations are used for python implementation.



$\frac{\partial \mathbf {J} }{ \partial \mathbf z^{[2]} } = \frac{1}{m} (\mathbf a^{[2]} - \mathbf y)$

$\frac{\partial \mathbf {J} }{ \partial W^{[2]} } = \frac{\partial \mathbf {J} }{ \partial \mathbf z^{[2]} } \mathbf a^{[1] T} $

$\frac{\partial \mathbf {J} }{ \partial b^{[2]} } = \frac{\partial \mathbf {J} }{ \partial \mathbf z^{[2]}}$

$\frac{ \partial \mathbf{J} }{ \partial z^{[1](i)} } =  W_2^T \frac{\partial \mathbf {J} }{ \partial z_{2}^{(i)} } * ( 1 - a^{[1] (i) 2})$

$\frac{\partial \mathbf {J} }{ \partial W_1 } = \frac{\partial J }{ \partial z_{1}^{(i)} }  X^T $

$\frac{\partial \mathbf {J} _i }{ \partial b_1 } = \sum{\frac{\partial \mathbf {J} }{ \partial z_{1}^{(i)}}}$
We will be using following notations python implementation:   

$dW1 = \frac{\partial J }{ \partial W_1 }$;

$db1 = \frac{\partial J }{ \partial b_1 }$;

$dW2 = \frac{\partial J }{ \partial W_2 }$;

$db2 = \frac{\partial J }{ \partial b_2 }$

[def]: 1