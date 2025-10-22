## Introducing Class and Object

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
## Layer Class Example

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

## Adding Attributes to a Class : Instance Variable
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


## Assigning Value to Instance Variable

Here we have added two attributes but the 'None' is assigned. We can assign the 'input' and 'output' using **attribute access operator**; 'dot' (.) as below:
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

## Add Parameters to Class Constructor

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

## Add a method to the class
A method in a class is a function that is defined within a class and operates on instances of that class. Methods are used to define behaviors and actions of the objects. We can also use methods to access and modify the attributes.

Methods can be categorized into **instance methods**, **class methods**, and **static methods**.

```js
class Layer:
  def __init__(self):
    self.input = None
    self.output = None

  def forward(s):
    # print(self, "Input Size : ", self.input, "| Output Size : ", self.output)
    print("Input Size : ", s.input, "| Output Size : ", s.output)
```

## Calling an Instance Method

We can call the method using class or object/instance. To call a method, the syntax is to use object name followed by ' . (dot) ', followed by method.


```js
L1=Layer()
L1.forward()        # Calling and instance method on instance/object
print(L1.input)
```

This will out put None

```js
L1=Layer(12288, 4)
L1.forward()        # Calling and instance method on instance/object
L1.input
```
This will throw an error that 1 expected 3 given. Let us try this

```js
L1=Layer()
L1.input=12288
L1.output=4
```

```js
L1.forward()
print(L1.input, type(L1.input))
```

## Calling an Instance Method on Class
Let us try to call a method on class.

```js
Layer.forward()
# Layer.forward(self)
```
Throws error in both the cases

We can call any instance method on class by using class name then dot (.) followed by method name and pass the instance of the class (object) as an arguement to the methods as below.

```js
Layer.forward(L1)
```

Till this point you are setting the attributes of the object by direct assignment after the class is instantiated or in other words, an object is created. That too, one at a time. There is better way of assigning attributes to the object. You can add the parameters in the __init__ method and then pass the required arguements during the instantiating.

```js
class Layer:
  def __init__(self, input, output):
    self.input = input
    self.output = output

  def forward(s):
    # print(self, "Input Size : ", self.input, "| Output Size : ", self.output)
    print("Input Size : ", s.input, "| Output Size : ", s.output)
```

```js
# create objects
L1=Layer()
# L1=Layer(12288, 4)
L1.forward()        # self is passes anonymously
Layer.forward(L1)   # Need Object if calling on layer for self
```   
Normally we call `L1.forward()` but Python actually converts it to `Layer.forward(L1)` thereby passing L1 as an argument to the parameter self (s in this case).

So both are equivalent!

One very common mistake about creating a method is that we miss to add 'self' as the first parameter of the method. Actually, when we call any methon on any instance or object of any class, the instance is automatically passed as an arguement with other arguements. So, it is must to use self in the method defination else an exception will occur.

## '_ _ repr _ _' method

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
L1 = Layer(12288, 4)
print(L1)
```

## Composition

Composition is a technique to add one or more objects from one or more classes to another class.

```js
class Weights:
  def __init__(self, w):
    self.wt = w
    
  def __repr__(self):
    return f"Weights(value={self.wt})"
```

Using an object of `Weights` class in the `Layer` Class

```js
class Layer:
  def __init__(self, input=None, output=None):
    self.input = input
    self.output = output

    # Composition
    self.weight = Weights(0.5)

  def __repr__(self):
    return f"Layer(input={self.input}, output={self.output}, Weights={self.weight}"

  def forward(self, x):
    self.input = x
    self.weighted = self.input * self.weight.wt
    return self.weighted
```

```js
L1 = Layer()
print(L1)   # Calls __repr__()

L1.forward(45)
print(L1)   # After forward pass
print(L1.weight)

```

`__repr__()` of Layer automatically calls `__repr__()` of Weights.

We see the whole composition tree when you print an object.

## Understanding '_ _ dict _ _'

'_ _ dict _ _' is an attribute of an object that stores all its attributes in a dictionary. This is a special attribute of an object and is automatically assigned. We can access this by using attribute access operator; 'dot' (.).

```js
class Layer:
  def __init__(self, input=None, output=None):
    self.input = input
    self.output = output

L1 = Layer(12288, 4)

print(L1.__dict__)
```
 
The output of this is a dictionary where the keys are attribute names (as strings) and the values are the corresponding attribute values.

You can also modify and/or add attributes dynamically using '_ _ dict _ _'.

```js
print("Before:", L1.__dict__)
# Modify an attribute using __dict__
L1.__dict__["input"] = 10
L1.__dict__["weights"] = Weights(0.9)

print("After:", L1.__dict__)
```

## Add a new attribute

You can add a new attribute to an object by assigning it into its __dict__, e.g., L1.__dict__["learning_rate"] = 0.01.
This makes learning_rate a valid attribute, accessible like L1.learning_rate.

```js
L1.__dict__["learning_rate"] = 0.01
print("After:", L1.__dict__)
print("Access new attribute:", L1.learning_rate)
``` 

## Class Variables

A class variable is a variable that is shared among all instances of a class. It is defined within a class but outside any instance methods or constructors (' _ _ init _ _' method). Class variables are used to store data that is common to all objects of the class.

```js
class Weights:
  def __init__(self, value):
    self.value = value

class Layer:
  learning_rate = 0.01 # Class Variable (shared across all layers)

  def __init__(self, input=None, output=None):
    # Instance variables (different for each object)
    self.input = input
    self.output = output
    self.weights = Weights(0.5)

  def forward(self, x):
    self.input = x
    self.weighted = self.input * self.weight.wt
    return self.weighted
```

```js
L1 = Layer()
print(L1.learning_rate)
print(Layer.learning_rate)
```

When you try to access the class variable from instance of a class, it will first check the instance attributes and when it does not find it there, it checks the attributes of the class the instance inherit from and use it.

You can modify the class variable (attribute) of an object of that class or update the class variable itself.

```js
# Modify `class` variable
print(L1.learning_rate)   # lr =0.01
L2 = Layer()
L2.learning_rate = 0.05
print(L2.learning_rate, L1.learning_rate)   # lr = 0.5 for L2 and 0.01 for L1
print(Layer.learning_rate)    # Update the lr for entire layer
Layer.learning_rate = 0.05
print(L2.learning_rate)
print(Layer.learning_rate)
print(L1.learning_rate)
```

## Class Method

We can add another functionality of counting how many objects we have created on Layer Class. As this functionality is not related to any of the object, we use special method named as Class Method for this task.

The class method is declared by using a decorator @classmethod.

As this method is not object dependent, so we will not use 'self' but instead, we have to use 'cls'.
Just to remind that the constructor or the ` __init__` method is only called by default when the object of the class is instantiated. So, we have to make some changes in the class.

We can also create one method in the Layer Class to count the Layers.

Here the 'total_layers' variable is also based on class, we call this variable as class variable and is initialized at the begining of the class. This is processed (incremented in this case) in the '_ _ init _ _ ' method.

```js
class Weights:
  def __init__(self, w):
    self.wt = w

  def __repr__(self):
    return f"Weights(value={self.wt})"

class Layer:
  learning_rate = 0.01
  total_layers = 0       # keeps track of how many Layer objects are created

  def __init__(self, input=None, output=None):
    self.input = input
    self.output = output
    Layer.total_layers += 1

    # Composition
    self.weight = Weights(0.5)

  def __repr__(self):
    return f"Layer(input={self.input}, output={self.output}, Weights={self.weight}"

  def forward(self, x):
    self.input = x
    self.weighted = self.input * self.weight.wt
    return self.weighted
        
  @classmethod
  def get_total_layers(cls):
    return cls.total_layers
```

Create an object of Layer class and find the number of Layers created. It is important to note that the method not called on the instance of the class but it is called on the class itself without instance reference.

```js
L1 = Layer()
print("Total Layers Created =", Layer.get_total_layers())
```   

## Creating Sub-Classes : Inheritance
Inheritance allows a class to inherit attributes and methods from another class. The class that inherits is called the "child class" or "subclass," and the class from which it inherits is called the "parent class" or "superclass."

The class that inherits from the parent class can add new properties and methods or override existing ones from the parent class.

Layer → Parent class / Superclass
In deep learning a layer is responsible for some computation on the basis of some input and produces some output. We have learnt that there are two computation in each forward and backward steps. The linear part is computed in the `DenseLayer` and activation part is computed in the `Activation` layer. 

We can now can create these classes on `Layer` class.
So, `Dense` Layer and `Activation` Layer are the Child class / Subclass (inherits from Layer).

We override the `forward()` method in DenseLayer to behave differently 
We can also define a different class variable in the subclass to show it can differ from the parent.

Let us update our `Layer` class to simplify the base class. It will only show the template of a class. All the functionalities you can add in the subclass or child class

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

Let us create a subclass to represent a Dense Layer

```js
class Dense(Layer):
  def __init__(self, input_size, output_size):
    self.weights = np.random.randn(output_size, input_size)
    self.bias = np.random.randn(output_size, 1)

  def forward(self, input):
    self.input = input
    return np.dot(self.weights, self.input) + self.bias
```


## Polymorphism
Polymorphism, an essential concept in OOP, allows the same interface to be used for different underlying forms (data types or objects). It can be categorized into two types: method overriding (runtime polymorphism) and operator overloading (compile-time polymorphism).

Method Overriding (Runtime Polymorphism):
Definition: Method overriding occurs when a subclass provides a specific implementation of a method that is already defined in its superclass. The subclass method overrides the superclass method, and the method call is resolved at runtime based on the object's actual type.  

In Python, polymorphism allows objects of different classes to be used interchangeably if they share the same method names.

We can demonstrate this for your Layer classes (Layer and DenseLayer) using a single function that calls forward() — the behavior will differ depending on the object type.  
