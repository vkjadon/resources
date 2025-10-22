## Google Colab

Google Colaboratory, also known as Colab! is a powerful cloud-based Python environment. It provides a free Python environment for running and developing Python code. It offers an interactive Jupyter notebook interface and allows you to access powerful hardware resources such as GPUs and TPUs.

Colab comes with many popular Python libraries pre-installed, such as NumPy, Pandas, and Matplotlib. You can start using them right away without worrying about installations.

Your Colab notebooks are automatically saved to your Google Drive. 

### Using Kaggle

1. Open kaggle.com
2. Click on the kaggle icon (upper right corner) adjacent to search input.
3. Click on the `Account`.
4. Scroll down to locale `API`.
5. Click on the "Create New API Tocken" button.
6. A file with name "kaggle.json" is downloaded

### Colab Side for Kaggle

1. Open Colab
2. pip install the kaggle `!pip install -q kaggle`
3. Click on the "Folder" icon (last item in the first left column of the icons).
4. Upload the Downloaded File using the upload icon. The first of the three icons appearing at top of the content tree.
5. Create directory `!mkdir ~/.kaggle`
6. Copy the uploaded file into the folder created in the previous step using `!cp kaggle.json ~/.kaggle/`.

```js
!pip install -q kaggle
```
In a Colab notebook, lines starting with ! are used to run shell commands.
`pip install -q` installs Python package in "quiet" mode suppressing the output during the installation process. `kaggle` is the Kaggle command-line tool, which allows you to interact with Kaggle's datasets and competitions from the command line.

```js
!mkdir ~/.kaggle
```
Upload the `kaggle.json` file to the current working directory using the "Folder" icon (last item in the first left column of the icons).
Now, upload the Downloaded File using the upload icon. The first of the three icons appearing at top of the content tree.

If you have successfully uploaded the `kaggle.json` in the current director, you can copy this to the `kaggle` directory created in the root and change the permissions.

```js
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
Download datset from kaggle using API of the dataset. You can use the following API to download.

```js
!kaggle datasets download -d ruchi798/housing-prices-in-metropolitan-areas-of-india
```

### Accessing Google Drive

1. Click on the Drive button in the Files top menu(the last of three icons).
2. This will add two lines of code to mount the drive.
3. Execute the code to mount.
4. Copy and Paste the code below if you are unable to locate the button.
`from google.colab import drive`
`drive.mount('/content/drive')`
5. A new directory(folder) with the name `drive` will appear.


## Handling `import`

In Python there are three method used to import a module.  
* **import**
* **from ... import ...**
* **from ... import** *

The import statements add the functionality to use functions and classes of the imported modules.

### Using `import module_name`
This statement imports the entire module, and you must reference the module name each time you use it.

```js
import math
number = 16
print("Math sqrt:", math.sqrt(number))
```

```js
import numpy as np
array = np.array([1, 4, 9, 16])
print("NumPy sqrt:", np.sqrt(array))
```
### Using `from ... import ...`

This statement imports specific attributes from the module, allowing you to use them directly without referencing the module name.

```js
from numpy import sqrt
print("NumPy sqrt:", sqrt(array))
```

```js
from math import sqrt
print("Math sqrt:", sqrt(number))
```
```js
from numpy import sqrt as np_sqrt
from math import sqrt as math_sqrt

print("Math sqrt:", sqrt(number))
print("NumPy sqrt:", np_sqrt(array))
```

### Using `from ... import *`

This imports all attributes from the module into the current namespace allowing you to use all the methods without referencing the module name.

```js
from math import *
from numpy import *

array = np.array([1, 4, 9, 16])
number = 16

print("Math sqrt:", sqrt(number))
print("Sqrt:", sqrt(array))
```
Change the order of the import statements and see the results

## Handling Error

Provides a robust way to handle exceptions and ensures your code can gracefully deal unexpected conditions without crashing.

Issues that occur at the compilation stage are errors.

Examples:  
* SyntaxError : Incorrect Python syntax.
* IndentationError : Incorrect indentation.

An exception is an event that disrupts the program's execution but can be handled programmatically.   

Examples:  
* ZeroDivisionError : Incorrect Python syntax.
* FileNotFoundError : Incorrect indentation.

```js
try:
  x=y
except Exception:
  print("An exception occurred")
```

Keeping the main logic (try) and exceptions handling (except) separate makes the code cleaner and easier to understand.

```js
try:
  x=y
except NameError:
  print("NameError exception occurred")
```
Another method to handle the exception is the use of default clean error message

```js
try:
  x=y
except Exception as error:
  print(error)
```
```js
test_file=open("test.txt")
x=9
```

This will throw `FileNotFoundError` error. We should use the following in such circumstances

```js
try:
  test_file=open("test.txt")
  x=y
except FileNotFoundError as error :
  print(error)
except Exception as error :
  print(error)
```

The finally block is used to ensure that resources are cleaned up properly, such as closing files or network connections, regardless of whether an exception was raised.

```js
try:
  test_file=open("test.txt")
  x=9
except FileNotFoundError as error :
  print(error)
except Exception as error :
  print(error)
else :
  print(test_file.read())
  test_file.close()
finally:
  print("Running Anyway ! Close all running tasks in this block")
```
## Understanding `__name__` variable

`__name__` is a special variable when we interact with modules and allows us to make distinction between modules we import (library, functions, classes etc) and module (python code) that we are currently executing.

```js
%%writefile module_first.py

def main():
    print(f"This is the main() function of First Module and the name of the First Module is {__name__} ")

if __name__=='__main__':
  main()

else:
    print("This is outside the main() function of First Module")
```
You can use `%%writefile filename.py` to create a `.py` file. The file created will be save in the session and will be earased once session is deleted.

Use `!python filename.py` to execute the `.py` file. When you run a module, a special variable `__name__` is created and assigned a name as `__main__`. 

So, when the file `module_first.py` is created, the `if` condition is satisfied and `main()` is called. That is why you get the output from the print statement from the `main()` method and the `else` is not executed.

Now, let us create another `.py` file named `module_second.py`

```js
%%writefile module_second.py

import module_first

print("Imported 'module_first' So, above is the output from import command")

def main():
    print(f"The name of the Second Module is {__name__} ")

if __name__=='__main__':
    main()
```



