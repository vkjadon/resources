## Numpy 

Deep Learning delas with huge data and NumPy is a very powerful and popular Python library. 

## Matrix Fundamentals

**NumPy stands for “Numerical Python”.**

This python library is suited for efficient array oriented computing.

The “numpy array” uses less memory and execution time than python list.

The internal type of numpy array is “ndarray”.

Consider an array $A_{m,n}$

$A_{m,n} =\begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n}
 \end{pmatrix}$

- The shape of this matrix is $m \times n$

- If we use $i, j$ as the indices to indicate an array element; then, for coding purposes $i$ will take $0$ to $m-1$ and $j$ will take $0$ to $n-1$.

## NumPy arrays using Python sequences

To initialize an array, we start by importing the NumPy library using `import numpy as np`

An array can be multi dimensional. We will start with 1D array. `np.array()` function is used to declare a numpy array. We can pass *List* or *Tuple* as an arguement to the `np.array()` function.

- a list of numbers will create a 1D array,

- a list of lists will create a 2D array,

- nested lists will create higher-dimensional arrays.

In general, any array object is called an ndarray in NumPy.

## 1D Array

We can use a tuple or list for creating a numpy array. Let us first pass a tuple of (1,2,3).

```js
array = np.array((1,2,3))
```
We can use `ndim` and `shape` methods on an array object to find the dimension and the shape of the numpy array.`shape` returns a tuple representing the size or exact number of elements of the array along each dimension. `ndim` return an integer representing the number of dimensions (axes) of the numpy array object.

```js
print(f"Array = {array} Dimension =  {array.ndim} Shape of the Array = {array.shape}")
```

Now, let us pass a list of [1,2,3] and print the array, dimension and the shape

```js
array = np.array([1,2,3])
```
Both these methods create a 1-dimensional NumPy array with shape (3,), containing three elements [1, 2, 3]. We can also see some other attributes such as type, size and dtype.

```js
print(f"Type of 1D Array array {type(array)}")
print(f"Size of 1D Array array {array.size}")
print(f"Type of data of elements of 1D Array array {array.dtype}")
```

## 2D Array

Consider an array $a_{i,j}$ representing a 2D array. A 2D array has two axes. Axis zero is along $i$ and axis one is along $j$. The given matrix has two rows and three columns.

We can create a 2D array with one row and three columns by passing one list in a list as an arguement of `np.array()` function.

```js
array = np.array([[1,2,3]])
print(f"Array = {array} Dimension =  {array.ndim} Shape of the Array = {array.shape}")
```
This method creates a 2D NumPy array with shape (1, 3), containing one row and three columns.

```js
print("Type of the Array : ", type(array))
print('Size of the Array = ', array.size)
print("Type of data of elements of the Array = ", array.dtype)
```
Now, we create another 2D array with two rows. We use `np.array([[2,3,1],[4,8,5]])` to initialize the array.

```js
array = np.array([[2,3,1],[4,8,5]])
print(f"Array = {array} Dimension =  {array.ndim} Shape of the Array = {array.shape}")
```

Two listss in a list are passed as an arguement in the np.array() function. We can see there are two arrays of size 3 in the outer array.

**First axis (i.e. Axis-0) has length of 2 and Second axis (i.e. Axis-1) has length of 3 (2L,3L)**

```js
print(f"Type of 1D Array array {type(array)}")
print(f"Size of 1D Array array {array.size}")
print(f"Type of data of elements of 1D Array array {array.dtype}")
```

## 3D Array

Consider an array $A$ representing a 3D array with its one element as $a_{i,j,k}$ . A 3D array has three axes. Axis zero is along $i$, axis one is along $j$ and axis 3 along $k$ axis. The given matrix has two arrays of shape $(3,4)$ so we can say that it has shape of $(2,3,4)$.

Let us now initialize this 3D array using `np.array()` function.
- The dimension of the array is 3 and the shape is $(2,3,4)$

```js
array=np.array([
            [[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]],
            [[12, 13, 14, 15],
              [16, 17, 18, 19],
              [20, 21, 22, 23]]
            ])
print(f"Array = {array} Dimension =  {array.ndim} Shape of the Array = {array.shape}")
```
**Axis-0 has length of 2, Axis-1 has length of 3 and Axis-2 each has length of 4**

```js
print(f"Type of 1D Array array {type(array)}")
print(f"Size of 1D Array array {array.size}")
print(f"Type of data of elements of 1D Array array {array.dtype}")
```

## Reshaping Matrices

We can reshape the array using the reshape() function in NumPy. Reshaping allows you to change the shape or dimensions of an array while maintaining the same number of elements. Otherwise, it will throw an error.

```js
array=array.reshape(8,3)
print(array)
```

We can specify one dimension by '-1' for the unknown shape. The Numpy will compute that on its own if exists.
```js
array.reshape(6,-1)
array.reshape(1,3,2,-1)
```
We can use `.T` to transpose the matrix.
```js
array.T
```
```js
train_x = np.random.randint(0, 255, size= (3, 2, 2, 2))
print(train_x)
```
Let us try to convert this matrix in the shape of (8, 3).

```js
train_set = train_x.reshape(3, -1).T
print(train_set)
```

The code below will also reshape the matrix in a shape of (8,3).

```js
train_set = train_x.reshape(-1, 3)
print(train_set)
```

You will find that both the output has different arrangements of the elements. This is because of in first case the reshape (before transpose) will flatten three rows of inner eight elements along rows and then transpose into $8 \times 3$ shpae. Whereas, in the second case, it will create eight rows taking inner three elemnts row wise.

## Indexing Array Elements
We can access arrays/elements of an array using indexing, just like in regular Python lists. For a 1D array first element is at index 0 and for 2D array first row at index `[0][]`, first column is at `[][0]` and first element is at index 0, 0 to specify row and column.

```js
array=np.arange(12).reshape(3,4)
print("2D Array : \n", array)
```
2D Array :


 [[ 0  1  2  3]  
 [ 4  5  6  7]  
 [ 8  9 10 11]]
 ```js
# Access elements of the array
print("First element:", array[0,0], "\nLast element:", array[1,-1])
```
First element: 0 

Last element: 7
```js
# Access rows of the array
print("First row at index 0] : ", array[0])
print("Last row : ", array[-1])
print("First column : ", array[:,:1])
```
```js
a=np.arange(18).reshape(3,2,3)
```
```js
a
```
```js
a[1][1][0]
```
```js
a=np.arange(36).reshape(2,3,2,3)
```
```js
a
```
```js
a[1]
```
## NumPy Array Creation Functions
## Creating 1D Array - arange()

```js
print(help(np.arange))
```
The * in the signature of numpy.zeros (or other Python functions) indicates that all arguments that come after the * must be passed as keyword arguments rather than positional arguments.

`arange([start,] stop[, step], [, dtype=None])`

**arange** returns evenly spaced values within a given interval starting from first optional parameter and within the `stop` value.

If only one parameter is given, it is assumed as `stop` and the `start` is automatically set to 0.

If the `step` is not given, it is set to `1` but if `step` is given, the `start` parameter cannot be optional, i.e. it has to be given as well.

The `step` sets the spacing between two adjacent values of the output array.

```js
np.arange(-3, 3, 0.5, dtype=int)
```
```js
dayTemperature=np.arange(6)
#Gives 6 values starting from 0 and default interval of 1
print(dayTemperature)
```
```js
dayTemperature=np.arange(22,30,1)
#First Value is 22 and it stops at 30, So 30 is not included
print(dayTemperature)
```
```js
dayTemperature=np.arange(25)
#First Value is 0 and it gives 25 output with default interval of 1
print(dayTemperature)
```
## Creating 1D Array - linspace()

`linspace(start, stop, num=50, endpoint=True, retstep=False)`

**Create NumPy array using linspacebar**

Array of linearly spaced values defined by `num` including `start` and `stop`

**default value of num is 50**

```js
dayTemperature=np.linspace(20,30,11)
print(dayTemperature)
```
[20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30.]
```js
dayTemperature, spacing=np.linspace(20,30, 20, endpoint=True, retstep=True)
print(dayTemperature, spacing)
```

## 2D Array - eye(): Identity Matrix

```
help(np.eye)
```
`np.eye(n, m)` defines a 2D identity matrix. The elements where i=j (row index and column index are equal) are 1 and the rest are 0. We can also pass only one argument, in that case a square identity matrix will be generated.

```js
print(np.eye(3))
```

array  
([[1., 0., 0.],  
 [0., 1., 0.],  
 [0., 0., 1.]])

```js
print(np.eye(3, 5))
```
array  
([[1., 0., 0., 0., 0.],    
 [0., 1., 0., 0., 0.],  
 [0., 0., 1., 0., 0.]])

## 2D Array - diag(): Diagonal Matrix

```
help(np.diag)
```
`np.diag` returns a square 2D diagonal array of the elements of given 1D array.

```js
print(array)
np.diag(array)
```
[[1 2]  
 [3 4]]    

array([1, 4])  

If we provide a 2D array, the `np.diag()` returns a 1D array with elements of the diagonal of the given 2D array.

```js
array = np.array([[1, 2], [3, 4]])
np.diag(array)
```
array([1, 4])
## ndarray - Random Functions

Use `np.random.rand` to create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).

```js
np.random.rand(4,2)
```
array  
([[0.4622304 , 0.89955765],  
 [0.97366722, 0.88483687],  
 [0.08845258, 0.24404886],  
 [0.7915377 , 0.86260121]])  

We can multiply by a factor if we wand higher magnitude. For example, if we need random number up to 255, we can do so as below:

```js
np.random.rand(4,2)*255
```
array  
([[222.10006886,  55.7687201 ],  
 [249.02645733, 136.95251295],  
 [237.03809814,   7.63058193],  
 [ 96.70100679,  17.769547  ]])  


Use `np.random.randn` to generates an array of given shape, filled with random floats sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1.

```js
help(np.random.randn)
import matplotlib.pyplot as plt

## Generate random samples

samples = np.random.randn(10000)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(samples, bins=50, color='g')
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
```
![alt text](image.png)
```js
array=np.random.randn(30,40)
# print(array)
np.mean(array)
```
0.006168837802542898  

```js
np.random.randint(3,6)
```
4
```js
np.random.randint(3,6, size=(2,4))
```
array  
([[3, 4, 3, 3],  
       [5, 5, 5, 4]]) 

```js
samples = np.random.randn(5000)
# print(samples)
np.mean(samples)
```
-0.033830634274922985

## Zeros, Ones and Full

```
print(help(np.zeros))
```
The * in the signature of numpy.zeros (or other Python functions) indicates that all arguments that come after the * must be passed as keyword arguments rather than positional arguments.

### Correct usage (passing arguments before * as positional or keyword)

```js
arr1 = np.zeros((4, 3))
arr2 = np.zeros((3, 3), order='F', dtype=int)
arr3 = np.zeros((3, 3), float)

### Correct usage (passing like as a keyword argument)

arr4 = np.zeros((2, 2), like=arr3, dtype=int)
# arr5 = np.zeros((3, 3), arr1)  # This will raise a TypeError

print(arr4)
```
[[0 0]  
 [0 0]]
```js
np.zeros((3,3,2))
```
array  
([[[0., 0.],  
        [0., 0.],  
        [0., 0.]],  
       [[0., 0.],  
        [0., 0.],  
        [0., 0.]],  
       [[0., 0.],
        [0., 0.],  
        [0., 0.]]])  
```js
np.ones((3,2,4))
```
array  
([[[1., 1., 1., 1.],  
        [1., 1., 1., 1.]],  
       [[1., 1., 1., 1.],  
       [1., 1., 1., 1.]],  
       [[1., 1., 1., 1.],  
        [1., 1., 1., 1.]]])
```js
np.full((2,3,4),4)
```
array  
([[[4, 4, 4, 4],  
        [4, 4, 4, 4],  
        [4, 4, 4, 4]],  
       [[4, 4, 4, 4],  
        [4, 4, 4, 4],   
        [4, 4, 4, 4]]])  
