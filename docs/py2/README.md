## Python Helping Function

There are many built-in functions in the python which helps us in better understanding the different methods available in python. I consider, `type()`, `dir()` and `help()` as some of the very important functions that must be used whenever you need some more clarification in understanding the use of any other methods.

### type() function

```js
integer_num=10
float_num = 40.5
string_model = 'HC-SR04'
```
```js
print(f'The type of `integer_num` is {type(integer_num)}')
print(f'The type of `float_num` is {type(float_num)}')
print(f'The type of `sensor_model` is {type(string_model)}')
```
Click [Formatted String](https://docs.python.org/3/tutorial/inputoutput.html) to go to documentation to explore more.

```js
dict = {"key-1" : "value-1", "key-2" : "value-2"}
print(type(dict))
dir(dict)
```
We see that `integer_num`, `float_num` and `string_model` belongs to `int`, `float` and the `str` classes respecyively. There many methods and attributes defined in a class. Let us use some of the methods of the `str` class.

```js
string_name = 'ultrasonic Sensor'
print(string_model + ' is a ' + string_name)
print(f"{string_model} is a {string_name}")
```
```js
if(str.islower(string_name)):
  print(f"{string_name} is in lower case")
else:
  print(string_name.upper())
  print(string_name)
```
We have used two different ways of using a method of a class. One way is to pass object through arguement of the class `str.islower(string_name)` method and other is using . (dot) operator on the object `string_name.upper()`.

```js
print(string_name.capitalize())
print(string_name.casefold())
print(string_name.center(20))
```
Click [String Class : str](https://docs.python.org/3/library/stdtypes.html#str) to go to documentation to explore more.

You may also be wondering how to identify the methods associated with the `str`. There exists a built-in function `dir()` in module `builtins` for this task.

### dir() and help() functions

The `dir()` function in Python is used to list the attributes and methods of an object, including special (or "magic") methods like `__init__` and `__str__`. When you pass an object to `dir()`, it returns a list of all the attributes and methods associated with that object.

```js
x=40.8
print(dir(x))
```


> [`'__abs__', '__add__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getformat__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__int__', '__le__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__pos__', '__pow__', '__radd__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rmod__', '__rmul__', '__round__', '__rpow__', '__rsub__', '__rtruediv__', '__setattr__', '__setformat__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__'`, 'as_integer_ratio', 'conjugate', 'fromhex', 'hex', 'imag', 'is_integer', 'real']

It gives a long list of special methods starting with double underscore, attributes and the methods. Let us find out some information about the `dir` using `help` function.

```js
help(dir)
```
Here, it is also important to note that the `dir` function returns a list of strings.

Coming back to the list returned by the `dir(x)`, for beginners, the methods and attribbutes are of initial interest not the special methods. You can use list comprehension to write logic which ignores the special methods.

```js
attributes_and_methods = [attr for attr in dir(x) if not attr.startswith('__')]
print(attributes_and_methods)
```

```js
string = 'abcd'
# attributes_and_methods = [strg for strg in dir(string) if not strg.startswith('__')]
print(help(str))
```
We will discuss list comprehension in later part of this session with some more examples. For now, the one line of the code is generating a new list of strings from the list of strings returned by the `dir()` built-in function of python.

You should also notice that the code is looping through each of the string inside the list and checking if it does not start with `__`.

Use `getattr(x, n)` when the attribute name is stored in a string (like when looping over `dir(x)`).

Use dot notation (`x.is_integer`) when you already know the attribute name in code.

```js
obj = 10

methods = [n for n in dir(obj) if not n.startswith('__') and callable(getattr(obj, n))]
attributes = [n for n in dir(obj) if not n.startswith('__') and not callable(getattr(obj, n))]

print(f"Attributes :  {attributes}")
print(f"Methods : {methods}")
```

```js
print(obj.real)        
print(obj.imag)        
print(obj.numerator)   
print(obj.denominator) 
```
For any integer obj in Python obj.imag is always 0 because integers are treated as a subset of complex numbers (n + 0j) and obj.denominator is always 1 because integers are treated as fractions (n/1).

```js
print(f"bit_length : {obj.bit_length()}")
print(f"bit_count : {obj.bit_count()}")
print(f"as_integer_ratio : {obj.as_integer_ratio()}")
print(f"conjugate :  {obj.conjugate()}")
print(f"to_bytes : {obj.to_bytes(2, 'big')}")
print(f"to_bytes : {obj.to_bytes(2, '')}")
```
Click [Integer/Float Class : int](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) to go to documentation to explore more.

You can check the details about `callable` and `getattr` by using `help()` function.

## Lambda Function

A lambda function, also known as an anonymous function, is a small and concise function in Python that doesn't require a defined name. It is defined using the lambda keyword, followed by the function arguements, a colon (:), and the expression to be evaluated.

Here are a few examples to illustrate the usage of lambda functions:

```js
def add(a,b):
  return a+b
sum=add(10,17)
print(sum)
```
```js
add = lambda a, b: a + b
```
This will create a function with name as 'add' with two parameters 'a' and 'b' and capable of performing the 'a+b' operation.

```js
result = add(3, 4)
print(result)  
```
```js
square = lambda x: x ** 2
result = square(5)
print(result)
```
It is not necessary to write the name of the function to pass the arguement. Rather, we can use right hand side with lambda keyword enclosed within parenthesis to act as function name and pass the arguements as below.

```js
(lambda a, b: a + b)(1,2)
```
The expression (lambda a, b: a + b)(1, 2) represents an Immediately Invoked Function Expression (IIFE) in Python. It creates an anonymous function that takes two arguments a and b and returns their sum. The function is then immediately invoked with the arguments 1 and 2.

```js
import numpy as np

# Define ReLU using lambda
relu = lambda x: np.maximum(0, x)

x = np.array([-2, -1, 0, 1, 2])
print("ReLU Output:", relu(x))
```

```js
losses = [("modelA", 0.32), ("modelB", 0.28), ("modelC", 0.45)]

# Sort models by loss (ascending)
sorted_losses = sorted(losses, key=lambda x: x[1])
print("Sorted Models:", sorted_losses)
```

### Keyword Arguements

```js
#Keyword arguement
expression = lambda a, b, c : a * (b + c)
#arguement can be in any order
result = expression(3, c=10, b=5)
# result = expression(c=10, 3, b=5) #Will throw an error
print(result)
```
### Default arguement

```js
#Default arguement
expression = lambda a, b, c=8 : a * (b + c)
# expression = lambda a, b=20, c : a * (b + c) #Will throw an error
result = expression(3, 10)
print(result)
```

## Comprehensions in Python

Comprehensions are a concise way to construct collections (list, set, dict, generator) from existing iterables.

### List Comprehension

```
[expression `for` item `in` iterable `if` condition]
```

Expression is used to produce the value of each item in the new list. The expression may contain if..else block. *for item in iterable* is used to loop over each item in an iterable (like a list, tuple, or range). *if condition* is optional which may be used to filter the items as per the condition when it is **True**.

```js
numbers = [1, 2, 3, 4, 5, 6]
```
Execute the following one at a time:

```js
print([x+2 for x in numbers])
print([x+2 for x in numbers if x % 2 == 0])
print(["odd" if x % 2 != 0 else "even" for x in numbers])
print(["even" for x in numbers if x % 2 == 0])
```

```js
data = list(range(20))  # pretend this is training data
batch_size = 5

batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
print("Batches:", batches)
```
```js
import numpy as np

image = np.random.randn(5, 28, 28, 3)

# Flatten images into 1D vectors
flattened = np.array([img.flatten() for img in image])
print("Flattened shape:", flattened.shape)
print("Flattened shape of first image:", flattened[0].shape)
```

### Dict Comprehension

```
{key_expression: value_expression for item in iterable if condition}
```
key_expression and value_expression are used to produce the key and the value of each item in the new list. The expression may contain if..else block. *for item in iterable* is used to loop over each item in an iterable (like a list, tuple, or range). *if condition* is optional which may be used to filter the items as per the condition when it is **True**.

Execute the following one at a time:

```js
print({x: x**2 for x in numbers})
print({x: "even" for x in numbers if x % 2 == 0})
print({x: "even" if x % 2 == 0 else "odd" for x in numbers})
```

```js
metrics = ["accuracy", "precision", "recall", "f1"]
values  = [0.91, 0.88, 0.84, 0.86]

results = {m: v for m, v in zip(metrics, values)}
print("Results Dict:", results)
```

### Set Comprehension

A set in Python is an unordered collection of unique elements.It is defined using { } or the set() constructor.No duplicates are allowed and the elements must be immutable (e.g., numbers, strings, tuples).

```js
labels = ["cat", "fish", "dog", "fish", "dog", "cat"]
unique_labels = set(labels)

for cls in unique_labels:
  print(f"Training classifier for class: {cls}")
```

### Generator Expression
Generator expressoin are handy especially when dealing with large datasets. You can load everything into memory rather than storing into memory.

```js
gen = (x*x for x in range(5))
# gen = [x*x for x in range(5)]
```

```js
print(gen)
print(next(gen))
```
```js
for val in gen:
  print(val)
```
After all values of the function are exhausted, its empty and can not be restarted. Saves memory (doesn't create the whole list in RAM). It is useful for streaming, large files, infinite sequences.

## Higher Order Function

A higher-order function is a function that can take one or more functions as arguments and/or return a function as its result.

Some common examples of higher-order functions in Python include map(), filter(), and reduce(). These functions can take other functions as arguments to perform operations on iterable objects.

Here's an example to illustrate the concept of a higher-order function:

```js
def double(x):
  return x * 2

def apply_operation(func, num):
  return func(num)

result = apply_operation(double, 5)
print(result)
```
In this example, `double` is a function that takes a number and returns its double. The `apply_operation` function is a higher-order function that takes a function *func* and a number *num*. It calls the provided function func with the given number num and returns the result.

We can pass the `double` function as an argument to `apply_operation` and provide a number (5 in this case). The `apply_operation` function will then call the double function with the provided number, resulting in 10, which is printed as the output.

**Higher-order functions** are powerful because they allow for code reuse, modularity, and the ability to abstract and manipulate behavior. They enable functional programming paradigms and can lead to more concise and expressive code.

```js
result = apply_operation(double, [6,9])
print(result)
```
[list] * 2 repeats the list.

To numerically double each element, you need a loop, comprehension, or map().

```js
def double(x):
  return [i * 2 for i in x]

result = apply_operation(double, [6, 9])
print(result)
```

```js
ho_func = lambda func, x : x + func(x)
ho_func(lambda x : x**2, 6)
```
First line of creates a function with name `func_ho` which require two arguements; one function, `func` and another `x`. The second line call this function and pass one function defined by `lambda` function and one arguement.

```js
ho_func(double, 6)
```

The `double` function doubles the number passed as arguement and `func_fo` add the number to the doubles number. So, output is three times the number.

### `map()` Function

**map()** - Applies a function to each element of an iterable and returns a new iterable with the transformed values. map produces a map object, which is an iterator and thus doesn't require generating a new list in memory. This is especially useful with large datasets as it avoids storing all intermediate results in memory at once.

```js
numbers = [1, 2, 3, 4, 5]
doubled_map = map(lambda x: x * 2, numbers)
print(doubled_map)         
print(next(doubled_map))
print(list(doubled_map))
print(list(doubled_map))         
```
map can apply a function across multiple iterables in parallel, which is something traditional for loops don’t handle as concisely.
For example, combining elements from two lists element-wise:

```js
list1 = [9, 2, 3, 6, 3]
list2 = [4, 5, 6, 8]
products = map(lambda x, y: x * y, list1, list2)

# Convert to list or directly iterate for further processing
for result in products:
    print("Product:", result)
```


### `filter()` Function

**filter()** - Filters elements from an iterable based on a condition defined by a function and returns a new iterable with the filtered values.

```js
numbers = [1, 2, 3, 4, 5]
even_numbers = filter(lambda x: x % 2 == 0, numbers)
# even_numbers = map(lambda x: x % 2 == 0, numbers)
print(list(even_numbers))
```

```js
def filter_number(x):
  # if x%2==0:
  if x > 3:
    return x
filtered_numbers = filter(filter_number, numbers)
# filtered_numbers = map(filter_number, numbers)
print(list(filtered_numbers))
```
### `reduce()` Function

**reduce()** - Applies a function to the elements of an iterable in a cumulative way and returns a single value.

```js
from functools import reduce
numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y + 2, numbers)
print(product)
print(help(reduce))
```
### `sorted()` Function

**sorted()** - Sorts the elements of an iterable based on a comparison defined by a function and returns a new list.

```js
fruits = ['apple', 'banana', 'cherry', 'durian']
sorted_fruits = sorted(fruits, key=lambda x: x[0])

print(sorted_fruits)
```
In this example, we have a list of tuples called data. We want to sort the list based on the second element of each tuple (i.e., the fruit name). We use a lambda function as the key parameter in the sorted() function, specifying that we want to sort based on x[1], which represents the second element of each tuple. The resulting sorted list is printed.

```js
data = [(2, 'Apple', 200), (3, 'Orange', 80), (1, 'Banana', 70)]
sorted_data = sorted(data, key=lambda x: x[0])
print(sorted_data)
help(sorted)
```
## * Operator

### Unpacking Operator *

The * operator can unpack a list or tuple of arguments directly into a function call.

```js
def add_three_numbers(a, b, c):
  return a + b + c

numbers = [1, 2, 3]
result = add_three_numbers(*numbers)
print(result)
```
`add_three_numbers(*numbers)`  is Equivalent to add_three_numbers(1, 2, 3).

The * operator can be used to unpack lists or tuples when combining them.

```js
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = [list1, *list2]
print(combined_list)
print(combined_list[2])
print(combined_list[0])
```
When you need to assign multiple variables from a list or tuple, * can capture remaining elements.

```js
values = [1, 2, 3, 4, 5, 6]
first, *middle, last = values
print(first)
print(middle)
print(last)
```
### *args

```js
def greet(*args):
  for name in args:
    print("Hello,", name)
greet("Satish", "Ramesh", "Kuldeep")
```

You can combine *args with regular parameters, allowing a function to accept required arguments and additional optional ones.

```js
def introduce(greeting, *args):
  print(greeting)
  for name in args:
    print(name)

introduce("Welcome everyone!", "Satish", "Ramesh", "Kuldeep")
```
### **kwargs

In Python, `**kwargs` (short for "keyword arguments") allows a function to accept an arbitrary number of keyword arguments. Also, the `*args` can be combined with both regular and keyword arguments. However, `*args` must come after all positional arguments and before `**kwargs` if used together.

```js
def display_data(title, *args, **kwargs):
    print("Title:", title)
    print("Items:", args)
    print("Details:", kwargs)

display_data("Summary", "Item1", "Item2", key1="value1", key2="value2")
```
In the following example, you can pass complete dictionary as keyword arguement:

```js
def introduce(name, age, city):
    print(f"My name is {name}, I am {age} years old and I live in {city}.")

info = {"name": "Kuldeep", "age": 30, "city": "New Delhi"}
introduce(**info)
```

 ## Iterable Helper

An iterable in Python is simply any object that can be "looped over" or "iterated through." In other words, it’s something you can use in a for loop to get each item one by one. It can be any collection of items, like a list, string, or even a range of numbers, where you go through each item in sequence. Technically speaking, the object should have a `__iter__` method in its class.

```js
my_list = [1, 2, 3, 4]

for item in my_list:
  print(item)

my_string = "hello"

for char in my_string:
  print(char)

for number in range(3):
  print(number)

my_dict = {"a": 1, "b": 2}
for key in my_dict:
  print(key)
```

You can check the object if the `__iter__` method exists or not using `dir` or any other piece of code.

### ZIP Function

The zip() function in Python is a built-in that lets you combine multiple iterables (lists, tuples, sets, ranges, etc.) element-wise into tuples. f the iterables are different lengths, zip stops at the shortest.

```js
list1 = [1, 2, 3, 5, 9, 78]
list2 = ["a", "b", "c", "d"]
combined = zip(list1, list2)
print(combined)
```
Run the below code *twice* and see the difference in both executions.

```js
for item in combined:
  print(item)
```
### Enumerate : `enumerate` Function

The enumerate() function in Python is very handy when you need both the index and the value while iterating over a sequence (like a list, tuple, or string).

losses = [0.9, 0.87,0.81, 0.76, 0.63, 0.69, 0.62, 0.58, 0.51, 0.45, 0.41, 0.38, 0.35]

```js
for step, loss in enumerate(losses, start=1):
  if step % 4 == 0:
    print(f"Step {step}, Loss={loss}")
```
```js
predictions = [0, 1, 0, 1, 1]
labels      = [0, 1, 1, 1, 0]

for inedx, (pred, label) in enumerate(zip(predictions, labels)):
  print(f"Sample {inedx}: Pred={pred}, True={label}")
```

```js
for inedx, (pred, label) in enumerate(zip(predictions, labels)):
  if pred != label: print(f"Sample {inedx}: Pred={pred}, True={label}")
```

### Range : `range` Function

In Python, `range` is a built-in function that generates a sequence of numbers. It is often used in loops (like for loops) when you want to repeat something a specific number of times or iterate over a sequence of integers.

```js
x = range(11, -5, -1)
# x = range(11, -5, -2)
for index in x:
  print(f"Index {index}, Value {x[index]}")
```
Now, you uncomment the second line and comment the first. See the results and try to find out the logic for the error thrown.

```js
epochs = 4
for epoch in range(epochs):
  print(f"Training epoch {epoch+1}")
```

```js
x = range(200, -100, -10)
for index in range(len(x)):  # looping over indices
  print(f"Index {index}, Value {x[index]}")
```

```js
x=range(40, -10, -10)
# z=list(x)
print(type(x))
y=iter(x)
print(y)
```
Execute `next(y)` in a separate cell.

### `any(), all(), min() and max()`

any(iterable) : Returns True if at least one element in the iterable is truthy, else False.

all(iterable) : Returns True if all elements in the iterable are truthy, else False.

min(iterable) : Returns the smallest element in the iterable.

max(iterable) : Returns the largest element in the iterable.

```js
nums = [0, 7, 5, 1]

print(any(nums))   
print(all(nums)) 

conditions = [x > 0 for x in nums]

print(conditions)  
print(any(conditions))  
print(all(conditions))
```

```js
students = [
    {"name": "S1", "score": 82},
    {"name": "S2", "score": 91},
    {"name": "S3", "score": 78}
]

topper = max(students, key=lambda s: s["score"])
print(topper) 
```
