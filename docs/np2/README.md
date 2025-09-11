## Matrix Operations - Element wise Operation

`multiply()`, `add()`, `subtract()` and `divide()`

- To perform element-wise multiplication, the dimension of both the arrays must be the same.
- The otherwise the broadcating can be used.

```js
#a=np.random.randn(4,3)
a=np.random.randn(3,2)
b=np.random.randn(3,2)
mult=np.multiply(a,b)
add=np.add(a,b)
sub=np.subtract(a,b)
div=np.divide(a,b)
```

### Arithmatic Functions

```js
np.sqrt(abs(np.exp(a)))
```
array  
([[1.85240584, 1.05251486],  
       [1.08180074, 0.63274813],  
       [1.02917014, 0.79802828]])
```js
np.log(abs(np.power(a,3)))
```
array  
([[ 0.62827889, -6.83763662],  
       [-5.54967852, -0.26529497],  
       [-8.56761968, -2.38738414]])  
```js
np.min(a)
```
-0.9153656687630972
```js
np.mean(a)
```
0.030584457697269585
```js
np.max(a)
```
1.232970496344989

## append() and insert()

```js
a = np.array([])
print("Empty Array:")
print(a)
print("Shape:", a.shape)
np.append(a,2)
```
Empty Array:  
[]  
Shape: (0,)  
array([2.])  
```js
a = np.array([])
y=np.arange(10)
for x in y:
  a=np.append(a, x)
print(a)
```
[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]  
```js
ones=np.ones(5)
zeros=np.zeros(5)
```
```js
np.append(ones,zeros)
```
```js
np.append(ones,[2,4])
```
```js
np.insert(ones, 3, [4,3])  
#inserted at the index specified
```

## Broadcasting

```js
print(a)
b=a+3
print(b)
b=b*7
print(b)
```
```js
a=np.random.rand(3,4)
b=np.random.rand(4,1)
c=a+b.T
print("Shape of a+b.T ",c.shape)
c=a.T+b
print("Shape of a.T+b ",c.shape)
```
```js
percent=100*a/np.sum(a)
print(percent)
```

```js
a = np.array([18, 2, 5, 3, 20]).reshape(5,1)
b = np.array([9, 14, 15, 3, -200]).reshape(5,1)
c=100*(abs(b-a))/a
print(c)
```

```js
a=np.arange(3).reshape(1,3)
b=np.arange(3).reshape(3,1)
dot=np.dot(a,b)
add=np.add(a,b)
```
```js
dot
```
```js
x=
```
```js
a.shape==(2,3)
b.shape==(2,3)
b.shape==(2,1)
b.shape==(1,3)
b.shape==(3,)
b.shape==(3,2)
b.shape==(2,)
```
```js

a.shape==(1000,256,256,256)
b.shape==(1000,256,256,256)
b.shape==(1000,1,256,256)
b.shape==(1000,1,1,256)
b.shape==(1,256,256,256)
b.shape==(1,256,256)
b.shape==(256,1)
b.shape==(1000,256,256)
```

## Dot Product
- To perform `dot product` on two muti-dimension arrays, the last dimension of the first array should be equal to last but one dimension of second array. So check `a.shape[-1]==b.shape[-2]`

Consider random arrays of shape `a(4,3)` and `b(3,2)`.
`np.dot(a, b)` has shape equal to number of rows of `a`, number of columns of `b`

- In numpy the `*` operator indicates element-wise multiplication and is different from `np.dot`.

- Dot Product is also known as inner product
```js
a=np.random.randint(1,10, size=(4,6))
b=np.random.randint(1,10, size=(6,2))
```
```js
print(a.shape[-1], b.shape[-2])
a.shape[-1] == b.shape[-2]
```
```js
if(a.shape[1] == b.shape[0]):
  c=np.dot(a,b)
  print(c, c.shape)
else:
  print("The columns and rows condition not satisfied")
  ```
  ```js
a=np.arange(10)
b=np.arange(10)
cDot=np.dot(a,b)
print(cDot)
```

## Slicing

**Colon Operator**.

Slicing is used to take desired elements of an array starting from one given index to another given index.

`[start : end: step]` for 1D array.

- **`start`** is inclusive and **`end`** is exclusive

- If we don't pass `start` its considered 0

- If we don't pass end its considered length of array in that dimension

- If we don't pass `step` its considered 1
```js
x = np.array([18, 2, 5, 3, 20, 4, 8, 18, 12])
```
```js
x[:]
```
array([18,  2,  5,  3, 20,  4,  8, 18, 12])
```js
x[3:] #All after index 3 including index 3
```
array([ 3, 20,  4,  8, 18, 12])
```js
x[:4] #All before index 4 excluding index 4.
```
array([18,  2,  5,  3])
```js
x[1:6]
```
array([ 2,  5,  3, 20,  4])
```js
x[1:7:2]
```
We use [`start` : `end`: `step` , `start` : `end`: `step` ] for 2D array.
```js
x=np.arange(20).reshape(4,5)
x
```
array  
([[ 0,  1,  2,  3,  4],  
       [ 5,  6,  7,  8,  9],  
       [10, 11, 12, 13, 14],  
       [15, 16, 17, 18, 19]])  

If we want [[5,6],[10,11]].

For this we will take
- 1:3 for the row
- 0:2 for the column
```js
x[1:3,0:2]
```
If we want [[12,13,14],[17,18,19]].

For this we will take
- 2:4 or 2: for the row
- 2:5 or 2: for the column
```js
x[2:4, 2:5]
```
array  
([[12, 13, 14],  
       [17, 18, 19]])  
```js
print("Slice of the array:", x[0:2]) #check Shape
print("Slice of the array:", x[1, 1:3])
print("Slice of the array:", x[2, 1:-1])
```
Slice of the array:   
[[0 1 2 3 4]  
 [5 6 7 8 9]]  
Slice of the array: [6 7]  
Slice of the array: [11 12 13]

**3D array Slicing**  
```js
x=np.arange(24).reshape(2,3,4)
x
```
array  
([[[ 0,  1,  2,  3],  
        [ 4,  5,  6,  7],  
        [ 8,  9, 10, 11]],  
       [[12, 13, 14, 15],  
        [16, 17, 18, 19],  
        [20, 21, 22, 23]]]) 

shape=2, 3, 4 means $i=2; j=3$ and $k=4$ where $i, j, k$ represent axis 0,1 and 2 respectively. There are two arrays of shape (3, 4) along axis 0.
If we want [[13,14],[17,18]].

For this we will take
- 1: for axis 0 ($i$) as we are working on second array (channel). There are two channels along axis 0.
- 0:2 for the row (axis 1 or $j$)
- 1:3 for the column (axis 2 or $k$)
```js
x[1:,0:2,1:3]
```

## Boolean- Indexing
```js
x>0.3
```
```js
x[x>0.3]
```
```js
np.where(x>0.3, x, 0)
```
## np.sum()
```js
a=np.arange(24).reshape(2,3,4)
```
```js
print(a)
```
[[[ 0  1  2  3]  
  [ 4  5  6  7]  
  [ 8  9 10 11]]  

 [[12 13 14 15]  
  [16 17 18 19]  
  [20 21 22 23]]]  
  ```js
a.sum() #output the sum of all the elements
```
- `sum(axis=0)` output a matrix of shape (3,4)
- `sum(axis=1)` output a matrix of shape (2,4)
- `sum(axis=2)` output a matrix of shape (2,3)
```js
a.sum(axis=0)
```
array  
([[12, 14, 16, 18],  
       [20, 22, 24, 26],  
       [28, 30, 32, 34]])  
 ```js
b=np.sum(a,axis=0, keepdims=True)
print(b)
b.shape
```
[[[12 14 16 18]  
  [20 22 24 26]  
  [28 30 32 34]]]  
(1, 3, 4)  
```js
b=np.sum(a,axis=1, keepdims=True)
print(b)
b.shape
```
[[[12 15 18 21]]  

 [[48 51 54 57]]]  

(2, 1, 4)  
```js
a.sum(axis=2)
```
The where parameter in numpy.sum allows you to specify a condition that determines which elements of the array should be included in the sum. Only the elements where the condition evaluates to True will be considered.
```js
a = np.array([1, -2, 3, -4, 5])
result = np.sum(a, where=(a > 0))
print(result)
```
9
```js
a = np.array([[10, 20, 30, 40, 50], [10, 20, 30, 40, 50]])
result = np.sum(a, axis=0, where=(a > 30))
print(result)
condition = np.array([True, False, True, False, True])
result = np.sum(a, where=condition)
print(result)
```
[  0   0   0  80 100]  
180  

## Stacking
```js  
f = np.array([1,2,3])
g = np.array([4,5,6])
print('Horizontal Append:', np.hstack((f, g)))
print('Vertical Append:', np.vstack((f, g)))
```
Horizontal Append: [1 2 3 4 5 6]  
Vertical Append: [[1 2 3]  
 [4 5 6]]  
 ```js
a = np.array([[1,1,2], [2,2,3], [4,3,3]])
b = np.array([[5,1,2], [6,2,3], [7,3,4]])
stack=np.vstack((a,b))
print('Vertical Stack', stack)
print(stack[5][0])
stack=np.hstack((a,b))
print('Vertical Stack', stack)
```
Vertical Stack [[1 1 2]   
 [2 2 3]  
 [4 3 3]  
 [5 1 2]  
 [6 2 3]  
 [7 3 4]]  
7    
Vertical Stack [[1 1 2 5 1 2]  
 [2 2 3 6 2 3]  
 [4 3 3 7 3 4]]  
 ```js
a=np.random.randn(4,3)
#a=np.random.randn(3,2)
b=np.random.randn(3,2)
c=a*b
print(c)
```
<pre style="background-color:#ffdddd; color:#b30000; padding:10px;">
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-4-e11d0d42acd0> in <cell line: 4>()
      2 #a=np.random.randn(3,2)
      3 b=np.random.randn(3,2)
----> 4 c=a*b
      5 print(c)

ValueError: operands could not be broadcast together with shapes (4,3) (3,2)
</pre>


<span style="color:blue;">To perform element-wise multiplication, the dimension of both the arrays must be the same.</span>

<span style="color:brown">The other method is broadcating.</span>
```js
a=np.arange(12).reshape(4,3)
b=np.arange(10,22).reshape(4,3)
c=a*b
print(c)
```
[[  0  11  24]  
 [ 39  56  75]  
 [ 96 119 144]  
 [171 200 231]]  
To perform `dot product` on two muti-dimension arrays, the last dimension of the first array should be equal to last but one dimension of second array. So check `a.shape[-1]==b.shape[-2]`
```js
print(a.shape[-1], b.shape[-2])
a.shape[-1] == b.shape[-2]
```
3 4  
False  
```js
a=np.random.randn(256,16)
b=np.random.randn(15,5)
if(a.shape[1] == b.shape[0]):
  c=np.dot(a,b)
  c.shape
  print(c)
else:
  print("The columns and rows condition not satisfied")
  ```
The columns and rows condition not satisfied    

```js
a=np.random.randn(3,4)
b=np.random.randn(4,1)
c=a+b.T
print("Shape of a+b.T ",c.shape)
c=a.T+b
print("Shape of a.T+b ",c.shape)
```
Shape of a+b.T  (3, 4)  
Shape of a.T+b  (4, 3)  
```js
a=np.random.randn(3,3)
b=np.random.randn(3,1)
c=a*b
print(c)
```
[[-0.17370979  0.0178696  -0.00774971]  
 [-0.0728755   0.21620951 -0.40034247]  
 [-0.47781078  0.26857552  0.36697574]]  
if x is a vector, then a Python operation such as $s = x + 3$ or $s = \frac{1}{x}$ will output s as a vector of the same size as x.
```js
print(a)
b=a+3
print(b)
b=b*7
print(b)
```
[[ 1.16499114 -0.11984312  0.05197373]  
 [-0.21990047  0.65240819 -1.208026  ]  
 [ 1.07587607 -0.60474562 -0.82631125]]  
[[4.16499114 2.88015688 3.05197373]  
 [2.78009953 3.65240819 1.791974  ]  
 [4.07587607 2.39525438 2.17368875]]  
[[29.154938   20.16109813 21.3638161 ]  
 [19.46069674 25.56685734 12.54381803]  
 [28.53113247 16.76678063 15.21582127]]  

Create an array corresponding to `(px_x*px_y*3)` and reshape to `(px_x, px_y,3)` and use `image.shape[0]`, `image.shape[1]` and `image.shape[2]`  

We can also use `image.reshape(-1,1)` for arranging all elements in a column vector. `-1` represnt unknown rows.  

We can also use `(1,-1)` for unknown columns and one row if required.
```js
a=np.random.randn(4*4*3)
image=a.reshape(4,4,3)
image.shape[0]
image.reshape(-1,1)
```
```js
import time

x1 = np.arange(10000)
x2 = np.arange(10000)

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print ("dot using for loop Computation time = " + str(1000 * (toc - tic)) + "ms")


### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot without loop Computation time = " + str(1000 * (toc - tic)) + "ms")
print(dot)
```
dot using for loop Computation time = 4.760352999999995ms  
dot without loop Computation time = 0.564473999999926ms  
333283335000  
```js
### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1), len(x2))) # we create a len(x1)*len(x2) matrix with only zeros

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i] * x2[j]
toc = time.process_time()
print ("outer = ----- Computation time = " + str(1000 * (toc - tic)) + "ms")
### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer =  ----- Computation time = " + str(1000 * (toc - tic)) + "ms")
#print(outer)
```
outer = ----- Computation time = 74992.794181ms  
outer =  ----- Computation time = 271.  8610609999814ms  
```js
### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
toc = time.process_time()
print ("elementwise multiplication  ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication ----- Computation time = " + str(1000*(toc - tic)) + "ms")
```
elementwise multiplication  ----- Computation time = 8.062335000005305ms  
elementwise multiplication ----- Computation time = 0.16506299999718976ms  
