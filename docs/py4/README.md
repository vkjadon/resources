# Introducing Class and Object

A **class** provides a **blueprint** or **template** for creating objects.  
It defines a set of attributes and methods for the objects that we create from the blueprint of a class.

Whereas, an **object** is an **instance of a class**.  
It represents a specific implementation of the class, with its own set of attributes and behaviors.

---

The class is just like designing a **form**.  
- The form contains some fields that are the same for every person/object.  
- The entries filled in the form are different for each individual.  

👉 Thus, the **form represents a class**, and the **fields represent the attributes**.  


---
```js
class Layer:
  pass
```
# Layer Class Example

This creates a `Layer` class but we have not added any fields (attributes) in that template of `Layer`.  
It is a common convention that the **class name starts with an uppercase**.


We can create object of this class by instantiating the class as below:

```js
L1=Layer()
L2=Layer()
```
```js    

print(L1)
print(L2)
```
<__main__.Layer object at 0x0000027b77529310>
<__main__.Layer object at 0x0000027b77529280>

This tells us that 'L1' and 'L2' are the instances of the Layer class, which is defined in the main module. The last values are the memory address where the objects are stored during execution. These are hexadecimal value and are unique to a particular instance.

As we create any object of a class by instantiating a class, that is why we also call the object as the instance of that class. The meaning remains the same. So, you should not be confused in case we write object or instance of a class.

# Adding Attributes to a Class : Instance Variable
Attributes are properties that belong to a class or instances of a class. They can be divided into two main categories of class variable and instance variable.
```js
class Layer:
    
    input = None
    output = None

L1=Layer()

print(L1.input)
print(L1.output)
```   
None  
None 

In the example of 'Layer' class, we have added two attributes of 'input' and 'output'. These are the instance variables and are accessible only through the specific instance. When we create any object using this class, these attributes are automatically assigned to the object. In this example, **None** is assigned to the *input* and *output*. These are accessed using **attribute access operator**; the 'dot' (.) on the object.


# Assigning Value to Instance Variable

Here we have added two attributes but the 'None' is assigned. We can assign the 'name' and 'model' using **attribute access operator**; 'dot' (.) as below:
```js
L1.input="Neural"
L1.output="Networks"
```
```js 
L2.input="Deep"
L2.output="Learning"
```
```js   
print(L1.input)
print(L2.output)
```
Neural  
Learning

We can assign the attribute using the this method but it is not very convenient when the attributes increases So, we use a function called constructor method to add attributes to a class. In python OOPS concepts, the functions are called as methods. So, during the discussion, we will be using method and function interchageably.

# Add Parameters to Class Constructor
In python, the constructor method is defined in ' _ _ init _ _ '. The first parameter in the constructor is 'self' which represent the instance of the class.

Add 'input' and 'output' as attributes of the Layer Class. In the context of the variable, the 'input' and 'output' are the attribute classified in the category of instance variable.
```js
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

L1=Layer()
L2=Layer()
L1.input="Neural"
L1.output="Networks"

L2.input="Deep"
L2.output="Learning"
  
print(L1.input)
print(L2.output)

```
Neural  
Learning

When we instantiate a class and create 'L1', the name of the object 'L1' is passed to the constructor **by default** and 'self' takes the name of the object. You can use any other variable to replace self, but it is a common practice in python to use 'self' to represent the instance or passing 'object name'

Also, we can access the Layer Class attributes from the outside of the class.

It is important to note the number of arguements we pass in the function call should be the same as the number of parameters we use to define a function.

# Add a method to the class
A method in a class is a function that is defined within a class and operates on instances of that class. Methods are used to define behaviors and actions of the objects. We can also use methods to access and modify the attributes.

Methods can be categorized into **instance methods**, **class methods**, and **static methods**.

```js
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = input
        return self.output
        

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        return f"Updated with learning_rate={learning_rate}, gradient={output_gradient}"

```
# Calling an Instance Method
We can call the method using class or object/instance. To call a method, the syntax is to use object name followed by ' . (dot) ', followed by method.

So method call is

**L1.forward("deep")**

```js
# Call forward
out = L1.forward("Deep Learning")
print("Forward Output:", out)       # Deep Learning
print("Stored Input:", L1.input)    # Deep Learning
print("Stored Output:", L1.output)  # Deep Learning

# Call backward
grad = L1.backward("Some Gradient", 0.01)
print("Backward Output:", grad)
```
     
Forward Output: Deep Learning  
Stored Input: Deep Learning  
Stored Output: Deep Learning  
Backward Output: Updated with learning_rate=0.01, gradient=Some Gradient 

# Calling an Instance Method on Class
Let us try to call a method on class.

```js
L1.forward()
L1.forward(self)
```

<div style="background-color: red; padding: 10px; border-radius: 8px;">
-----------------------------------------------------------------------------
NameError<br>
Traceback (most recent call last):
  File "C:\Users\HP\Desktop\Deep Learning\class.py",   line 29, in <module>
    L1.forward(self,"deep")  
               ^^^^
NameError: name 'self' is not defined 
</div>

Throws error in both the cases

We can call any instance method on class. In this case we use class name then dot (.) followed by method name and pass the instance of the class (object) as an arguement to the methods as below.


```js
class Layer:
    def __init__(self, name):
        self.name = name

class Network:
    def add_layer(self, layer_obj):
        print("Adding:", layer_obj.name)

# create objects
L1 = Layer("Input Layer")
L2 = Layer("Output Layer")

net = Network()
net.add_layer(L1)   # Adding: Input Layer
net.add_layer(L2)   # Adding: Output Layer

```   
Normally we call:
```js
L1.forward("Neural")
```

But Python actually converts it to:
```js
Layer.forward(L1, "Neural")
```
Here, L1 is passed as an argument to the parameter self.

So both are equivalent!

One very common mistake about creating a method is that we miss to add 'self' as the first parameter of the method. Actually, when we call any methon on any instance or object of any class, the instance is automatically passed as an arguement with other arguements. So, it is must to use self in the method defination else an exception will occur.

# '_ _ repr _ _' method
Python default display for an object is the class and memory location. You can customize this output using user-defined '_ _ repr _ _()' method.

```js
class Layer:
    def __init__(self, input=None, output=None):
        self.input = input
        self.output = output

    def __repr__(self):
      return f"Layer(input={self.input}   output={self.output})"
```
```js
L1 = Layer("Neural", "Networks")
L2 = Layer("Deep", "Learning")

print(L1)
print(L2)
```
Layer(input=Neural, output=Networks)  
Layer(input=Deep, output=Learning)

# Composition
Composition is a technique to add one or more objects from one or more classes to another class.
```js
class Weights:
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f"Weights(value={self.value})"

class Activation:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"Activation(name='{self.name}')"

class Layer:
    def __init__(self, input=None, output=None):
        self.input = input
        self.output = output

        # Composition
        self.weights = Weights(0.5)
        self.activation = Activation("ReLU")

    def __repr__(self):
        return (f"Layer(input={self.input}, output={self.output}, "
                f"weights={self.weights}, activation={self.activation})")

    def forward(self, x):
        self.input = x
        weighted = f"{x}*{self.weights.value}"
        self.output = self.activation.name + "(" + weighted + ")"
        return self.output

```
```js
L1 = Layer()
print(L1)   # Calls __repr__()

L1.forward("Neural Input")
print(L1)   # After forward pass

```
Layer(input=None, output=None, weights=Weights(value=0.5), activation=Activation(name='ReLU')) 
Layer(input=Neural Input, output=ReLU(Neural Input*0.5), weights=Weights(value=0.5), activation=Activation(name='ReLU'))

__repr__() of Layer automatically calls __repr__() of Weights and Activation.

We see the whole composition tree when you print an object.

# Understanding '_ _ dict _ _'
'_ _ dict _ _' is an attribute of an object that stores all its attributes in a dictionary. This is a special attribute of an object and is automatically assigned. We can access this by using attribute access operator; 'dot' (.).

```js
class Layer:
    def __init__(self, input=None, output=None):
        self.input = input
        self.output = output

L1 = Layer("Neural", "Networks")

print(L1.__dict__)

```
{'input': 'Neural', 'output': 'Networks'}

```js
L1.activation = "ReLU"
print(L1.__dict__)
```
{'input': 'Neural', 'output': 'Networks', 'activation': 'ReLU'}
```js
print(Layer.__dict__)   # Shows methods, docstring, etc.
```
{'__module__': '__main__', '__init__': <function Layer.__init__ at 0x000001b4b4d093a0>, 'forward': <function Layer.forward at 0x000001b4b4d094e0>, 'backward': <function Layer.backward at 0x000001b4b4d09580>, '__dict__': <attribute '__dict__' of 'Layer' objects>, '__weakref__': <attribute '__weakref__' of 'Layer' objects>, '__doc__': None}  

The output of this is a dictionary where the keys are attribute names (as strings) and the values are the corresponding attribute values.

We can also modify and/or add attributes dynamically using '_ _ dict _ _'.

We can also modify and/or add attributes dynamically using '_ _ dict _ _'.


```js
print("Before:", layer)
# Modify an attribute using __dict__
L1.__dict__["input"] = 10
L1.__dict__["weights"] = Weights(0.9)
L1.__dict__["activation"] = Activation("Sigmoid")

print("After:", layer)
```
```js
print("Forward result:", L1.forward(5))
```
Before: Layer(input=None, output=None, weights=Weights(value=0.5), activation=Activation(name=ReLU))  
After: Layer(input=10, output=None, weights=Weights(value=0.9), activation=Activation(name=Sigmoid))  
Forward result: Sigmoid(5*0.9)

```js
print("Before:", L1.__dict__)

# Add a new attribute
L1.__dict__["learning_rate"] = 0.01

print("After:", L1.__dict__)
print("Access new attribute:", L1.learning_rate)
```
Before: {'input': None, 'output': None, 'weights': Weights(value=0.5), 'activation': Activation(name=ReLU)}  
After: {'input': None, 'output': None, 'weights': Weights(value=0.5), 'activation': Activation(name=ReLU), 'learning_rate': 0.01}
Access new attribute: 0.01  

You can add a new attribute to an object by assigning it into its __dict__, e.g., L1.__dict__["learning_rate"] = 0.01.
This makes learning_rate a valid attribute, accessible like L1.learning_rate.

# Class Variables
A class variable is a variable that is shared among all instances of a class. It is defined within a class but outside any instance methods or constructors (' _ _ init _ _' method). Class variables are used to store data that is common to all objects of the class.

```js
class Weights:
    def __init__(self, value):
        self.value = value


class Activation:
    def __init__(self, name):
        self.name = name


class Layer:
    # ---- Class Variable (shared across all layers) ----
    default_learning_rate = 0.01

    def __init__(self, input=None, output=None):
        # Instance variables (different for each object)
        self.input = input
        self.output = output
        self.weights = Weights(0.5)
        self.activation = Activation("ReLU")

    def layer_info(self):
        print("Input :", self.input, "| Output :", self.output,
              "| Weights :", self.weights.value, "| Activation :", self.activation.name)

    def forward(self, x):
        self.input = x
        weighted = f"{x}*{self.weights.value}"
        self.output = f"{self.activation.name}({weighted})"
        return self.output

    def get_learning_rate(self):
        # Access class variable using self or class name
        print(f"Default Learning Rate = {self.default_learning_rate}")
        # print(f"Default Learning Rate = {Layer.default_learning_rate}")
```
```js
l1 = Layer()
l2 = Layer()

l1.layer_info()
l1.get_learning_rate()

# Modify class variable
Layer.default_learning_rate = 0.05
l2.get_learning_rate()
```
Input : None | Output : None | Weights : 0.5 | Activation : ReLU  
Default Learning Rate = 0.01  
Default Learning Rate = 0.05  

default_learning_rate is a class variable → same for all Layer objects.   
input, output, weights, and activation are instance variables → unique per object.  
When we try to access the class variable from instance of a class, it will first check the instance attributes and when it does not find it there, it checks the attributes of the class the instance inherit from and use it.

# Class Method
We can add another functionality of counting how many objects we have created on Layer Class. As this functionality is not related to any of the object, we use special method named as Class Method for this task.

The class method is declared by using a decorator @classmethod.

As this method is not object dependent, so we will not use 'self' but instead, we have to use 'cls'.
Just to remind that the constructor or the ' _ _ init _ _ ' method is only called by default when the object of the class is instantiated. So, we have to make some changes in the class.

We can also create one method in the Layer Class to count the Robots.

Here the 'total_layers' variable is also based on class, we call this variable as class variable and is initialized at the begining of the class. This is processed (incremented in this case) in the '_ _ init _ _ ' method.

```js
class Weights:
    def __init__(self, value):
        self.value = value


class Activation:
    def __init__(self, name):
        self.name = name


class Layer:
    # ---- Class variables ----
    total_layers = 0       # keeps track of how many Layer objects are created
    base_layer_cost = 5000 # shared by all Layer objects

    def __init__(self, input=None, output=None):
        # Instance variables
        self.input = input
        self.output = output
        self.weights = Weights(0.5)
        self.activation = Activation("ReLU")

        # Increment the class variable when a new Layer is created
        Layer.total_layers += 1

    def layer_info(self):
        print("Input :", self.input, 
              "| Output :", self.output,
              "| Weights :", self.weights.value, 
              "| Activation :", self.activation.name)

    def forward(self, x):
        self.input = x
        weighted = f"{x}*{self.weights.value}"
        self.output = f"{self.activation.name}({weighted})"
        return self.output

    def get_base_cost(self):
        print(f"The Base Cost of a Layer is = {self.base_layer_cost}")

    # ---- Class method ----
    @classmethod
    def get_total_layers(cls):
        return cls.total_layers
```
```js
l1 = Layer()
l2 = Layer()
l3 = Layer()

l1.layer_info()
l1.get_base_cost()

print("Total Layers Created =", Layer.get_total_layers())
```
Input : None | Output : None | Weights : 0.5 | Activation : ReLU  
The Base Cost of a Layer is = 5000  
Total Layers Created = 3  

total_layers → class variable (shared by all Layer objects).  
base_layer_cost → another class variable.  
get_total_layers() → class method to access total_layers.  
input, output, weights, and activation → instance variables (unique for each Layer).   

# Encapsulation
Encapsulation refers to the bundling of attributes and methods into a class. Encapsulation also restricts direct access to some of an object's components and hides the internal representation of an object from the outside. This is achieved using access modifiers.

Now let us try to divide the Layer class into two classes to encapsulate the configuration attributes and related methods into one class and the other behavioral methods into another class.

First, let us define our configuration class
```js
class Weights:
    def __init__(self, value):
        self.value = value


class Activation:
    def __init__(self, name):
        self.name = name


class LayerConfig:
    """
    Encapsulates configuration attributes for a neural network layer.
    """
    def __init__(self, input=None, output=None, weight_value=0.5, activation_name="ReLU"):
        # Private attributes (note the underscore convention)
        self._input = input
        self._output = output
        self._weights = Weights(weight_value)
        self._activation = Activation(activation_name)

    # Getter and Setter for input
    def get_input(self):
        return self._input

    def set_input(self, value):
        self._input = value

    # Getter and Setter for output
    def get_output(self):
        return self._output

    def set_output(self, value):
        self._output = value

    # Access weights
    def get_weights(self):
        return self._weights

    # Access activation
    def get_activation(self):
        return self._activation

    def __repr__(self):
        return (f"LayerConfig(input={self._input}, output={self._output}, "
                f"weights={self._weights.value}, activation={self._activation.name})")
```
This LayerConfig class encapsulates all config-related attributes and provides controlled access through getter and setter methods.

In Python, prefixing an attribute with an underscore (e.g., _name) signals that it is intended to be private and shouldn’t be accessed directly from outside the class. Instead, we provide getter (and optionally setter) methods.

Here’s how we can modify the LayerConfig class to introduce a private _name attribute along with a get_name() method:
```js
class LayerConfig:
    """
    Encapsulates configuration attributes for a neural network layer.
    Demonstrates private attribute and getter/setter methods.
    """
    def __init__(self, name, input=None, output=None, weight_value=0.5, activation_name="ReLU"):
        self._name = name                 # private attribute
        self._input = input
        self._output = output
        self._weights = Weights(weight_value)
        self._activation = Activation(activation_name)

    # ---- Getter for name ----
    def get_name(self):
        return self._name

    # ---- Setter for name ---- (optional, if you want to allow changes)
    def set_name(self, new_name):
        self._name = new_name

    # ---- Getter & Setter for input ----
    def get_input(self):
        return self._input

    def set_input(self, value):
        self._input = value

    # ---- Getter & Setter for output ----
    def get_output(self):
        return self._output

    def set_output(self, value):
        self._output = value

    # ---- Access Weights & Activation ----
    def get_weights(self):
        return self._weights

    def get_activation(self):
        return self._activation

    def __repr__(self):
        return (f"LayerConfig(name={self._name}, input={self._input}, "
                f"output={self._output}, weights={self._weights.value}, "
                f"activation={self._activation.name})")
```
```js
config = LayerConfig("HiddenLayer1")

# Direct access discouraged: config._name
print("Name (using getter):", config.get_name())

# Change name using setter
config.set_name("HiddenLayerX")
print("Updated Name:", config.get_name())
```
Name (using getter): HiddenLayer1  
Updated Name: HiddenLayerX  

Layer class will focus on behavior (like forward) and will use the LayerConfig object internally, hiding its attributes from direct access.

```js
class Layer:
    """
    Behavioral class for a neural network layer.
    Uses LayerConfig internally to encapsulate configuration.
    """
    def __init__(self, config: LayerConfig):
        # Composition: store config internally (private)
        self._config = config

    def forward(self, x):
        # Update input in config
        self._config.set_input(x)
        weighted = f"{x}*{self._config.get_weights().value}"
        output = f"{self._config.get_activation().name}({weighted})"
        self._config.set_output(output)
        return output

    def layer_info(self):
        # Access config attributes via getters
        print(f"Layer Name: {self._config.get_name()}")
        print(f"Input: {self._config.get_input()}")
        print(f"Output: {self._config.get_output()}")
        print(f"Weights: {self._config.get_weights().value}")
        print(f"Activation: {self._config.get_activation().name}")

    def get_config(self):
        # Optional: allow access to the config object if needed
        return self._config

    def __repr__(self):
        return f"Layer(name={self._config.get_name()}, activation={self._config.get_activation().name})"
```
```js
# Create config with private name
config1 = LayerConfig("HiddenLayer1", weight_value=0.8, activation_name="Sigmoid")

# Create Layer object using config
layer1 = Layer(config1)

# Check info before forward pass
layer1.layer_info()

# Perform forward pass
print("Forward Output:", layer1.forward(5))

# Check info after forward pass
layer1.layer_info()

# Representation
print(layer1)
```
Layer Name: HiddenLayer1  
Input: None  
Output: None  
Weights: 0.8  
Activation: Sigmoid  
Forward Output: Sigmoid(5*0.8)  
Layer Name: HiddenLayer1  
Input: 5  
Output: Sigmoid(5*0.8)  
Weights: 0.8  
Activation: Sigmoid  
Layer(name=HiddenLayer1, activation=Sigmoid)  

_config is private, so the internal configuration is hidden from outside.  
Access to attributes like name, input, weights is only via getter/setter methods.  
Behavioral methods (forward, layer_info) operate on _config without exposing internals.

# Creating Sub-Classes : Inheritance
Inheritance allows a class to inherit attributes and methods from another class. The class that inherits is called the "child class" or "subclass," and the class from which it inherits is called the "parent class" or "superclass."

The class that inherits from the parent class can add new properties and methods or override existing ones from the parent class.

Layer → Parent class / Superclass
DenseLayer → Child class / Subclass (inherits from Layer)
We override the forward() method in DenseLayer to behave differently 
We also define a different class variable (base_layer_cost) in the subclass to show it can differ from the parent.

```js
class Layer:
    # Class variable
    base_layer_cost = 5000

    def __init__(self, name, weight_value=0.5, activation_name="ReLU"):
        self._name = name                # private attribute
        self.weights = weight_value
        self.activation = activation_name

    def forward(self, x):
        output = f"{self.activation}({x}*{self.weights})"
        return output

    def get_name(self):
        return self._name

    def get_base_cost(self):
        print(f"Base cost of Layer: {Layer.base_layer_cost}")

    def __repr__(self):
        return f"Layer(name={self._name}, activation={self.activation})"


# Subclass
class DenseLayer(Layer):
    # Different base cost for DenseLayer
    base_layer_cost = 8000

    # Override forward method
    def forward(self, x):
        # Suppose DenseLayer applies an extra step (like bias addition)
        output = f"{self.activation}(({x}*{self.weights}) + bias)"
        return output

    def get_base_cost(self):
        print(f"Base cost of DenseLayer: {DenseLayer.base_layer_cost}")
```
```js
layer1 = Layer("HiddenLayer1", weight_value=0.6, activation_name="Sigmoid")
dense_layer1 = DenseLayer("DenseLayer1", weight_value=0.9, activation_name="Tanh")

print(layer1)
print("Forward output (Layer):", layer1.forward(5))
layer1.get_base_cost()

print(dense_layer1)
print("Forward output (DenseLayer):", dense_layer1.forward(5))
dense_layer1.get_base_cost()
```
Layer(name=HiddenLayer1, activation=Sigmoid)  
Forward output (Layer): Sigmoid(5*0.6)  
Base cost of Layer: 5000  
Layer(name=DenseLayer1, activation=Tanh)  
Forward output (DenseLayer): Tanh((5*0.9) + bias)  
Base cost of DenseLayer: 8000  

DenseLayer inherits from Layer → shows inheritance.  
forward() in DenseLayer overrides the parent method → shows method overriding.  
base_layer_cost is different in subclass → demonstrates class variable overriding.  

# Polymorphism
Polymorphism, an essential concept in OOP, allows the same interface to be used for different underlying forms (data types or objects). It can be categorized into two types: method overriding (runtime polymorphism) and operator overloading (compile-time polymorphism).

Method Overriding (Runtime Polymorphism):
Definition: Method overriding occurs when a subclass provides a specific implementation of a method that is already defined in its superclass. The subclass method overrides the superclass method, and the method call is resolved at runtime based on the object's actual type.  

In Python, polymorphism allows objects of different classes to be used interchangeably if they share the same method names.

We can demonstrate this for your Layer classes (Layer and DenseLayer) using a single function that calls forward() — the behavior will differ depending on the object type.  

```js
# Supporting classes
class Weights:
    def __init__(self, value):
        self.value = value

class Activation:
    def __init__(self, name):
        self.name = name

# Encapsulated configuration class
class LayerConfig:
    def __init__(self, name, weight_value=0.5, activation_name="ReLU"):
        self._name = name          # private attribute
        self._weights = Weights(weight_value)
        self._activation = Activation(activation_name)
        self._input = None
        self._output = None

    # Getter and setter for name
    def get_name(self):
        return self._name

    def set_name(self, new_name):
        self._name = new_name

    # Getter and setter for input/output
    def get_input(self):
        return self._input

    def set_input(self, x):
        self._input = x

    def get_output(self):
        return self._output

    def set_output(self, y):
        self._output = y

    # Access weights and activation
    def get_weights(self):
        return self._weights

    def get_activation(self):
        return self._activation

    def __repr__(self):
        return f"LayerConfig(name={self._name}, activation={self._activation.name})"

```
```js
# Parent Layer class
class Layer:
    base_layer_cost = 5000

    def __init__(self, config: LayerConfig):
        self._config = config

    def forward(self, x):
        self._config.set_input(x)
        output = f"{self._config.get_activation().name}({x}*{self._config.get_weights().value})"
        self._config.set_output(output)
        return output

    def __repr__(self):
        return f"Layer(name={self._config.get_name()}, activation={self._config.get_activation().name})"
```
```js

# Subclass DenseLayer
class DenseLayer(Layer):
    base_layer_cost = 8000

    def forward(self, x):
        self._config.set_input(x)
        # Override: apply extra step for DenseLayer (like bias)
        output = f"{self._config.get_activation().name}(({x}*{self._config.get_weights().value}) + bias)"
        self._config.set_output(output)
        return output
```
```js
# Polymorphism demonstration function
def run_forward(layer_obj, input_value):
    """
    Accepts any Layer object (Layer or DenseLayer)
    and calls its forward method.
    """
    print(f"{layer_obj} -> Forward Output: {layer_obj.forward(input_value)}")

```
```js
# --- Usage ---
config1 = LayerConfig("HiddenLayer1", weight_value=0.6, activation_name="Sigmoid")
config2 = LayerConfig("DenseLayer1", weight_value=0.9, activation_name="Tanh")

layer1 = Layer(config1)
dense_layer1 = DenseLayer(config2)

run_forward(layer1, 5)
run_forward(dense_layer1, 5)

```
Layer(name=HiddenLayer1, activation=Sigmoid) -> Forward Output: Sigmoid(5*0.6)  
Layer(name=DenseLayer1, activation=Tanh) -> Forward Output: Tanh((5*0.9) + bias)
 
_config is private, demonstrating encapsulation.
DenseLayer overrides forward() → method overriding.
run_forward() works with both Layer and DenseLayer → polymorphism.
The internal configuration is hidden from outside, only accessed via getters/setters.

## Operator Overloading (Compile-Time Polymorphism):

Operator overloading allows the same operator to have different meanings based on the context. In Python, this is achieved by defining special methods like add, sub, mul, etc.

Layer class — for example, we can overload the + operator to combine two layers’ weights.

```js
class Layer:
    def __init__(self, name, weight_value=0.5, activation_name="ReLU"):
        self._name = name
        self.weights = weight_value
        self.activation = activation_name

    def forward(self, x):
        return f"{self.activation}({x}*{self.weights})"

    def __repr__(self):
        return f"Layer(name={self._name}, weights={self.weights}, activation={self.activation})"

    # Operator Overloading: +
    def __add__(self, other):
        if isinstance(other, Layer):
            # Combine weights for demonstration
            combined_weights = self.weights + other.weights
            return Layer(name=f"{self._name}+{other._name}",
                         weight_value=combined_weights,
                         activation_name=self.activation)
        else:
            raise TypeError("Operand must be a Layer object")
```
```js
layer1 = Layer("Layer1", weight_value=0.5, activation_name="ReLU")
layer2 = Layer("Layer2", weight_value=0.7, activation_name="ReLU")

# Add two Layer objects using overloaded + operator
layer3 = layer1 + layer2

print(layer1)
print(layer2)
print("Combined Layer:", layer3)
print("Forward output of combined layer:", layer3.forward(5))
```
Layer(name=Layer1, weights=0.5, activation=ReLU)  
Layer(name=Layer2, weights=0.7, activation=ReLU)  
Combined Layer: Layer(name=Layer1+Layer2, weights=1.2, activation=ReLU)  
Forward output of combined layer: ReLU(5*1.2)  


__add__() is a special method for +.

Operator overloading allows custom behavior for +, here we combined weights of two layers.

This is compile-time polymorphism — the meaning of + depends on the object type.



