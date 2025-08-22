## How to use Python Help

The dir() function in Python is used to list the attributes and methods of an object, including special (or "magic") methods like __init__ and __str__. When you pass an object to dir(), it returns a list of all the attributes and methods associated with that object.

In a simple assignment statement x=10, x is also object of a class. So, we can use dir() function of this as well as below:

```js
x=40.8
print(dir(x))
```
> [`'__abs__', '__add__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getformat__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__int__', '__le__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__pos__', '__pow__', '__radd__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rmod__', '__rmul__', '__round__', '__rpow__', '__rsub__', '__rtruediv__', '__setattr__', '__setformat__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__'`, 'as_integer_ratio', 'conjugate', 'fromhex', 'hex', 'imag', 'is_integer', 'real']

It gives a long list of special methods starting with double underscore, attributes and the methods. For beginners, the methods and attribbutes are of initial interest. You can use lint comprehension to write logic which ignores the special methods.

```js
attributes_and_methods = [attr for attr in dir(x) if not attr.startswith('__')]
print(attributes_and_methods)
```

## Lambda Function

A lambda function, also known as an anonymous function, is a small and concise function in Python that doesn't require a defined name. It is defined using the lambda keyword, followed by the function arguements, a colon (:), and the expression to be evaluated.

Here are a few examples to illustrate the usage of lambda functions: