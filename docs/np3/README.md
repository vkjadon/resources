## Handling Image Data (3D Matrix)

## Create Image

Color images are represented by 3D array corresponding to pixels in height, pixels in width for each of RGB channel.

```js
randA=np.random.rand(64*64*3)*255
```

reshape to a row or column matrix as below with three channels for RGB.  
```js
#Use this for Tensor Flow
image=randA.reshape(64,64,3)
#Use this for OpenCV
# image=randA.reshape(3,64,64)
image = image.astype('uint8')
```

This represent an color image of $64 \times 64$ and is stored in array of size (64, 64, 3). The last 3 represent the R, G and B channels. If you observe the `Image` matrix, you can see there are three elements in one/inner array. There are 64 such lines of arrays. These 64 lines of arrays of 3 elemets are further appearing 64 times to complete the `Image` matrix. In `Deep Learning` applications handling images, we call each pixel values as features and need to be converted to a column vector.
```js
image
```
ndarray (64, 64, 3)  
![  ](image-1.png)  
array  
([[[184,   2, 150],  
        [204,  73, 162],  
        [156,  68, 161],  
        ...,  
        [241, 227, 215],  
        [216, 122, 170],  
        [151,  40, 199]],  
       [[ 58, 180, 172],  
        [ 31, 192, 253],  
        [167, 147, 121],  
        ...,  
        [108, 181, 103],  
        [ 95, 113, 230],  
        [136, 253,  91]],  
         [[191, 128, 240],  
        [ 82, 136, 152],  
        [165, 169,  62],  
        ...,  
        [227, 235,  12],  
        [211, 173, 134],  
        [230, 180,  30]],  
       ...,
       [[ 10,  26, 143],  
        [ 95, 186, 156],  
        [ 48,  95, 173],  
        ...,  
        [ 26, 245,  75],  
        [  3, 173, 244],  
        [243, 109,  35]],  
....  
       [[201,  94,  63],  
        [216, 120,  18],  
        [132, 138,  12],  
        ...,  
        [150,  26, 141],  
        [186,  77, 173],  
        [ 16, 172, 178]],  
...  
       [[ 17,  56,  53],  
        [129,   1,  89],  
        [ 54, 136,  26],  
        ...,  
        [115,  50, 240],  
        [ 42, 156,  51],  
        [237,  39, 204]]],  
         dtype=uint8)

For (64, 64, 3) (channel-last format):

Colab recognizes this as a typical RGB image format, assuming the first two dimensions represent the image's height and width. The third dimension represents the color channels.
Since the data type is uint8 (a common type for image data with values from 0 to 255), Colab renders it as an image.

For (3, 64, 64) (channel-first format):

Colab doesn't recognize this shape as a standard image format. Instead it interprets the array as a generic 3D array and displays the raw data (a representation of the array values) rather than rendering it as an image. Colab does not assume that the first dimension represents channels.
```js
image[0][0].shape #Check shape
```
(3,)  
```js
image[0][0] #Check shape
```
array([224, 156, 168], dtype=uint8)
```js
image[0][0][0] #Check shape
```
184
```js
image.reshape(-1,1)
```
array  
([[184],  
       [  2],  
       [150],  
       ...,  
       [237],  
       [ 39],  
       [204]],   dtype=uint8)  
```js
image.reshape(1,-1)
```
array([[184,   2, 150, ..., 237,  39, 204]], dtype=uint8)  
```js
image 
```
ndarray (64, 64, 3)  
![  ](image-2.png)  
array  
([[[184,   2, 150],  
        [204,  73, 162],  
        [156,  68, 161],  
        ...,  
        [241, 227, 215],  
        [216, 122, 170],  
        [151,  40, 199]],  
  ...  
       [[ 58, 180, 172],  
        [ 31, 192, 253],  
        [167, 147, 121],  
        ...,  
        [108, 181, 103],  
        [ 95, 113, 230],  
        [136, 253,  91]],  
  ... 
       [[191, 128, 240],  
        [ 82, 136, 152],  
        [165, 169,  62],  
        ...,  
        [227, 235,  12],  
        [211, 173, 134],  
        [230, 180,  30]],  
        ...,  
        [[ 10,  26, 143],  
        [ 95, 186, 156],  
        [ 48,  95, 173],  
        ...,  
        [ 26, 245,  75],  
        [  3, 173, 244],  
        [243, 109,  35]],  
...  
       [[201,  94,  63],  
        [216, 120,  18],  
        [132, 138,  12],  
        ...,  
        [150,  26, 141],  
        [186,  77, 173],  
        [ 16, 172, 178]],  
...  
       [[ 17,  56,  53],  
        [129,   1,  89],  
        [ 54, 136,  26],  
        ...,  
        [115,  50, 240],  
        [ 42, 156,  51],  
        [237,  39, 204]]], dtype=uint8)  

## Plot NumPy Array Image using Matplotlib
```js
import matplotlib.pyplot as plt
```
```js
imgplot = plt.imshow(image)
```
![  ](image-3.png)
In the above example, we created a dummy image but in deep learning we are given with the images and we may require to render these images. We can use PIL library to import the image. Let us first import an image into our session from button available at top left corner of the 'File/Folder' menu. This uploded image will be destroyed once we close the session.

## Convert Image Format into Numpy Array
```js
from PIL import Image

# Open the image using PIL
image = Image.open('resize_test_1.png')

# Convert the image to a NumPy array
image_array = np.array(image)

# Display the shape of the array
print("Image array shape:", image_array.shape)

imgplot = plt.imshow(image_array)
```
<pre style="background-color:#ffdddd; color:#b30000; padding:10px;">
FileNotFoundError                         Traceback (most recent call last)
<ipython-input-12-b82d58061b14> in <cell line: 4>()
----> 4 image = Image.open('resize_test_1.png')

FileNotFoundError: [Errno 2] No such file or directory: '/content/resize_test_1.png'
</pre>

We observe that the images have four channels instead of the expected three channels for RGB (Red, Green, Blue). The additional channel is typically an alpha channel, which represents transparency information.

The presence of an alpha channel indicates that the image has an alpha transparency layer, commonly used in formats like PNG that support transparency. The alpha channel stores transparency information for each pixel, indicating how opaque or transparent the pixel should be when rendered.

If we want to work with the RGB channels only, we can convert the image to RGB mode using the convert() function in PIL.
```js
# Convert the image to RGB mode
image_rgb = image.convert('RGB')

# Convert the image to a NumPy array
image_array = np.array(image_rgb)

# Display the shape of the array
print("Image array shape:", image_array.shape)

imgplot = plt.imshow(image_array)
```
Image array shape: (475, 200, 3)  
![  ](image-4.png)  
```js
import cv2 as cv2
```
```js
#Input the Resolution
rows=int(input('Enter the Height of the Viewer Window: '))
cols=int(input('\nEnter the Width of the Viewer Window: '))
```
Enter the Height of the Viewer Window: 64  

Enter the Width of the Viewer Window: 64  
```js
#Input the R-G-B
print('\n**Note: Please enter the brightness values between(0-255)**')
red_value=int(input('\nEnter the Brightness Value of RED Channel: '))
green_value=int(input('\nEnter the Brightness Value of GREEN Channel: '))
blue_value=int(input('\nEnter the Brightness Value of BLUE Channel: '))
```
**Note: Please enter the brightness values between(0-255)**

Enter the Brightness Value of RED Channel: 50

Enter the Brightness Value of GREEN Channel: 100

Enter the Brightness Value of BLUE Channel: 150  
```js
#Matrix Generation of R-G-B
red_channel = np.uint8(np.ones([rows,cols])*red_value)
blue_channel = np.uint8(np.ones([rows,cols])*blue_value)
green_channel = np.uint8(np.ones([rows,cols])*green_value)

#Color Maker
color = cv2.merge([red_channel,green_channel,blue_channel])
plt.imshow(color)
print(color.shape)
```
(64, 64, 3)  
![alt text](image-5.png)  
```js
from google.colab.patches import cv2_imshow
# Display the image
cv2_imshow(image)
``` 
![alt text](image-6.png)  
```js
# Merge channels into an image
image_bgr = cv2.merge((blue_channel, green_channel, red_channel))  # BGR format (OpenCV default)

# Display with OpenCV (BGR format is correct here)
cv2_imshow(image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# If we display the same image in Matplotlib without conversion, colors will be incorrect
plt.imshow(image_bgr)
plt.title("Incorrect Display in Matplotlib")
plt.show()

# Convert BGR to RGB for correct Matplotlib display
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Correct display with Matplotlib
plt.imshow(image_rgb)
plt.title("Correct Display in Matplotlib")
plt.show()
```
![alt text](image-7.png)  
![alt text](image-8.png)  
![alt text](image-9.png)  


