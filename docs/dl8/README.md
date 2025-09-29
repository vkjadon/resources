## Layer class

```js
class Layer:
  def __init__(self):
    self.input = None
    self.output = None

  def forward(self, input):
    pass

  def backward(self, output_gradient, learning_rate):
    pass
```

The Layer class serves as the base class for all layers in the neural network.
It has two instance variables: *input* and *output*, initialized to None.

**Forward Method**

The `forward` method takes an *input* and is meant to perform the `forward` pass through the layer.

However, in this base class, the forward method does nothing (indicated by the pass statement).

It is expected to be overridden by a subclasses.

**Bacward Method**

The backward method takes *output_gradient* (the gradient of the loss with respect to the layer's output) and *learning_rate* as arguments.

Again, in this base class, the backward method does nothing and is expected to be overridden by a subclasses.

## Dense Layer

Every `Dense` layer has associated weights and biases. The weights and biases are initialized in the `__init__()` method on the basis of the sizes of the layer called *output_size* and the size of the previous layer called the *input_size*.

The layer also takes input from the previous layer and produces an output for the forward propagation. This is achieved in the `forward()` method. The method takes *input* as the *arguement* and was initialized in the parent class `__init__()` method.

The layer is also responsible for computing the gradient of the cost with respect to parameters (weights and biases) for given gradient of the cost with respect to the output. It also need to compute the cost gradient with respect to input for passing to the previous layer for computing these expressions. These are required to update the weights and biases for the gradient descent.

Let us create a subclass 'Dense` on base class `Layer` as below:

```js

class Dense(Layer):
  def __init__(self, input_size, output_size):
    np.random.seed(10)
    self.weights = np.random.randn(output_size, input_size)
    self.bias = np.random.randn(output_size, 1)

  def forward(self, input):
    self.input = input
    return np.dot(self.weights, self.input) + self.bias

  def backward(self, output_gradient, learning_rate):
    weights_gradient = np.dot(output_gradient, self.input.T)
    input_gradient = np.dot(self.weights.T, output_gradient)
    self.weights -= learning_rate * weights_gradient
    self.bias -= learning_rate * output_gradient
    return input_gradient

```
**Check Random Weights**

```js
Dense(2,3).weights
Dense(3,3).bias
```
Changing the `seed()` statement outside the `__init__()` method may be above it or even in the `__init__()` method of the parent class will not have the effect of generating the same number.

**Dense class (subclass of Layer)**

The `Dense` class represents a fully connected layer in a neural network, where each neuron in the layer is connected to every neuron in the previous layer.

It inherits from the `Layer` class, which means it inherits the properties and methods of the base class.

The `__init__()` method is defined to initialize the Dense layer. The method takes two arguments: *input_size* and *output_size* representing the number of neurons in the previous layer and the current layer, respectively.

Inside the `__init__()` method, two instance variables are initialized: weights and bias.

The weights variable is a randomly initialized matrix of shape (*output_size*, *input_size*).

Each element of the matrix represents the weight associated with the connection between the neurons of the previous layer and the neurons of the current layer.

The bias variable is a randomly initialized matrix of shape (*output_size*, 1).

It represents the bias term associated with each neuron in the current layer.

**backward method**

The backward method overrides the backward method of the base class.

It takes two arguments *output_gradient* and *learning_rate*. *output_gradient* represents the gradient of the loss with respect to the *output* of the 'Dense' layer.

The method performs the backward pass by calculating the gradient of the loss with respect to the *weights* and *input* of the `Dense` layer.

The gradient of the loss with respect to the *weights* is computed by taking the dot product of *output_gradient* and the transpose of *self.input*.

The gradient of the loss with respect to the *input* is computed by taking the dot product of the transpose of *self.weights* and *output_gradient*.

The *weights* and *bias* are updated using gradient descent: the *weights* are updated by subtracting the product of *learning_rate* and *weights_gradient*, and the *bias* is updated by subtracting the product of *learning_rate* and *output_gradient*.

The method returns the gradient of the loss with respect to the input, which will be passed to the previous layer in the backward pass.

Overall, the `Dense` class represents a fully connected layer in a neural network. It initializes random weights and biases, performs the forward pass by calculating the dot product of the weights and input, and computes the gradients during the backward pass for parameters updates and further backpropagation.

The output of the dense layer is passes through the activation layer which produces an output in the forward propagation. The activation layer is also responsible for computing the cost gradients with respect to the input of the activation layer on the basis of the output gradients. There is no trainable parameters in the activation layer.

There are various activation functions inuse in the deep learning. We can generalize the activation layer by initializing the activation function in the `__init__()` method. The activation function is used in the `forward` method to compute the activations.

Let is create an activation class as below:

```js

class Activation(Layer):
  def __init__(self, activation, activation_prime):
    self.activation = activation
    self.activation_prime = activation_prime

  def forward(self, input):
    self.input = input
    return self.activation(self.input)

  def backward(self, output_gradient, learning_rate):
    return np.multiply(output_gradient, self.activation_prime(self.input))

```
**Activation class (subclass of Layer)**

The `Activation` class represents an activation layer that applies a specific activation function to its input.

It inherits from the Layer class, meaning it has access to the input and output variables as well as the forward and backward methods defined in the base class.

The `__init__()` method takes two arguments: *activation* and *activation_prime* representing the activation function and its derivative, respectively.

The forward method overrides the base class method. It takes an input, assigns it to the input variable, applies the activation function (self.activation) to the input, and returns the result.

The backward method also overrides the base class method. It takes output_gradient and learning_rate as arguments.

It calculates the gradient of the loss with respect to the input of the activation layer by multiplying output_gradient with the derivative of the activation function evaluated at the input (self.activation_prime(self.input)).

The result is then returned.

Now we are ready to create the classes for different activation functions. Let us start with `Tanh` class based on the `Activation` class. Just to recall that the `Activation` class is subclass of the `Layer` base class.

```js
class Tanh(Activation):
  def __init__(self):
    def tanh(x):
      return np.tanh(x)

    def tanh_prime(x):
      return 1 - np.tanh(x) ** 2

    super().__init__(tanh, tanh_prime)
```
**Tanh class (subclass of Activation):**

The Tanh class inherits from the Activation class, which means it inherits the properties and methods of the base class (`Layer`) as well as the `Activation` class.

The `__init__()` method is defined to initialize the Tanh activation layer.
Inside the `__init__()` method, two inner functions, tanh and tanh_prime, are defined.

The `tanh` function takes an input x and applies the hyperbolic tangent function (np.tanh) to it. This function returns the tanh of x.

The `tanh_prime` function takes an input x and calculates the derivative of the hyperbolic tangent function, which is $1 - (np.tanh(x))^2$. This function returns the derivative of tanh with respect to x.

Finally, the `super().__init__` line calls the constructor of the superclass (Activation) and passes the tanh and tanh_prime functions as arguments. This initializes the Activation superclass with the tanh activation function and its derivative.

By creating the Tanh class, the code extends the functionality of the Activation class by providing a specific activation function (tanh) and its derivative. This allows the Tanh activation layer to be used in a neural network for non-linear transformations during the forward and backward passes.

```js
class Sigmoid(Activation):
  def __init__(self):
    def sigmoid(x):
      return 1 / (1 + np.exp(-x))

    def sigmoid_prime(x):
      s = sigmoid(x)
      return s * (1 - s)

    super().__init__(sigmoid, sigmoid_prime)
```

## Example of the Activation Layer

Let us understand the passing the function to the Super Layer of the `Activation` Layer. In the following example, we will create a `Square` subclass on the `Activation` class and implement the Square Function thrugh the super class. We can pass the two inner functions defined in `__init__` method through the `super().__init__` method. This initilaizes the superclass `Activation` or These functions are set as instance variables in the `__init__` method of `Activation` class.

```js

class Square(Activation):
  def __init__(self):
    def square(x):
      return x*x

    def square_prime(x):
      return 1

    super().__init__(square, square_prime)
```

```js

Layer=Square()
Layer.forward(2)
Layer.backward(5,0.07)

```

```js
def mse(y_true, y_pred):
  return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
  return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
  return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
  return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
```