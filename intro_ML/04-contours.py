# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cv2
import matplotlib.pyplot as plt
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import sobel
import os
from scipy import ndimage
from IPython import get_ipython

# %% [markdown]
# # Practice on images descriptors (except deep descriptors)
# ## M1 DSAI - Intro to ML
# ### Diane Lingrand (Diane.Lingrand@univ-cotedazur.fr)

# %%
from skimage import io
from skimage import data
from skimage import transform
import numpy as np
from matplotlib import pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## Images from the library scikit-image

# %%
#img = data.coffee()
img = data.chelsea()
# img = io.imread('/home/lingrand/Ens/SSII/Cours8-contours/carreNoir.png') #data.coffee()
# you can save an image to an image file
# io.imsave("coffee.png",img)


# %%
print(img.shape)


# %%
plt.imshow(img)


# %%
# a figure with 2 images
fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes.ravel()
ax[0].imshow(data.coffee())
ax[0].title.set_text('a cup of coffee')
ax[1].imshow(data.chelsea())
ax[1].title.set_text('a cat')

# %% [markdown]
# ## Smoothing using convolution
# %% [markdown]
# We will start to use a simple method of 2D convolution: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html">scipy.ndimage.convolve</a>.
#
# Gaussian smoothing is approximated by a convolution with the kernel $\frac{1}{16}\begin{pmatrix} 1 & 2 & 1\\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{pmatrix}$.
#
# This convolution only deals with a single channel. We thus need to apply this function on each channel and then recompose another image.

# %%


# %%
lissGauss3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
r = ndimage.convolve(img[:, :, 0], lissGauss3x3)
g = ndimage.convolve(img[:, :, 1], lissGauss3x3)
b = ndimage.convolve(img[:, :, 2], lissGauss3x3)
imgLisse = np.dstack((r, g, b))
plt.imshow(imgLisse)

# %% [markdown]
# Strange colors ... What's the problem ? How to improve ?

# %%
lissGauss3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
r = ndimage.convolve(img[:, :, 0], lissGauss3x3)
g = ndimage.convolve(img[:, :, 1], lissGauss3x3)
b = ndimage.convolve(img[:, :, 2], lissGauss3x3)
imgLisse = np.dstack((r, g, b))
plt.imshow(imgLisse)

# %% [markdown]
# This image looks better !
# %% [markdown]
# ## Edges by first derivative
# %% [markdown]
# A well-know detector is the one by Sobel. Have a look on the edges long x-axis and y-axis before composing the final result.

# %%
sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/4.0
sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/4.0

r = ndimage.convolve(img[:, :, 0], sobelx)
g = ndimage.convolve(img[:, :, 1], sobelx)
b = ndimage.convolve(img[:, :, 2], sobelx)
imgSobelx = np.dstack((r, g, b))
plt.imshow(imgSobelx)
print("min = ", imgSobelx.min())
print("max = ", r.max())
print(imgSobelx[100, 400, :])

# %% [markdown]
# If you are note sure of the result, try your detector on a simple image composed of a black square on a white background.

# %%
img = io.imread(os.getcwd() + '/black_square.png')
plt.imshow(img)


# %%
sobel_mag = ndimage.sobel(img[:, :, 0], axis=0)
print(type(sobel_mag), sobel_mag.shape, type(
    sobel_mag[0][0]), sobel_mag.min(), sobel_mag.max())
plt.imshow(sobel_mag, cmap=plt.cm.gray)
# ou axis =1
plt.imshow(np.abs(ndimage.sobel(img[:, :, 0], axis=0)), cmap=plt.cm.gray)

# %% [markdown]
# Why do you obtain only a single horizontal edge ?
# %% [markdown]
# And what about to write the code from scratch ?

# %%
img[:, :, 0].shape


# %%
# img has only a single channel and k is from odd dimensions.
# we will ignore borders
def maConvolution(img, k):
    (h, w) = img.shape
    dimK = k.shape[0]
    d = dimK//2
    res = np.zeros(shape=(h, w), dtype=np.float64)
    for i in range(d, h-d):
        for j in range(d, w-d):
            res[i][j] = np.mean(img[i-d:i+d+1, j-d:j+d+1]*k)
    return res


sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/4.0
sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/4.0

r = np.abs(maConvolution(img[:, :, 0], sobelx))
g = np.abs(maConvolution(img[:, :, 1], sobelx))
b = np.abs(maConvolution(img[:, :, 2], sobelx))
imgSobelx = np.dstack((r, g, b))
plt.imshow(imgSobelx)
print("min = ", imgSobelx.min())
print("max = ", r.max())
print(imgSobelx[100, 100, :])

# %% [markdown]
# Let's forger ndimage for Sobel edges and take a look to scikit-image.

# %%


# %%
sobelx = sobel(img, axis=1)  # or axis = 0
print(sobelx.min(), sobelx.max())
plt.imshow(np.abs(sobelx))
print(type(sobelx[0][0][0]))


# %%
# we now compute the strengh of the edges
sobel_mag = np.sqrt(sobel(img, axis=0)**2 + sobel(img, axis=1)**2)/math.sqrt(2)
plt.imshow(sobel_mag)

# %% [markdown]
# Apply Sobel edges detection to several images of your choice.R

# %%
# for you
img = data.coffee()
print(img[:, :, 1].shape)
plt.imshow(img)
sobel_mag = np.sqrt(sobel(img[:, :, 1], axis=0) **
                    2 + sobel(img[:, :, 1], axis=1)**2)/math.sqrt(2)

# %% [markdown]
# Test also othe edge detectors using first derivatives such as, for example, <a href="https://scikit-image.org/docs/dev/api/skimage.filters.html?highlight=sobel#skimage.filters.prewitt">Prewitt</a>.

# %%
# for you

# %% [markdown]
# ## Edges by second derivative
# %% [markdown]
# Test also the detector from <a href="https://scikit-image.org/docs/dev/api/skimage.filters.html?highlight=sobel#skimage.filters.laplace">Laplace</a>.

# %%
# for you

# %% [markdown]
# ## Edges by difference of gaussians
# %% [markdown]
# Smooth a image twice using different smoothin and look at the difference of the results (absolute value).

# %%
# for you

# %% [markdown]
# ## Thresholding of the edges points
# %% [markdown]
# Test simple thresholding or <a href="https://scikit-image.org/docs/dev/api/skimage.filters.html?highlight=sobel#skimage.filters.apply_hysteresis_threshold">hysteresis</a>. Is it obvious to find the correct threshold value?

# %%
# for you

# %% [markdown]
# ## Points of interest
# %% [markdown]
# ### Harris
# %% [markdown]
# Let's start with Harris:

# %%


# %%
pts = corner_peaks(corner_harris(img[:, :, 1]), min_distance=1)
print(pts.shape[0], ' points found')
plt.imshow(img, cmap='gray')
plt.scatter(y=pts[:, 0], x=pts[:, 1], c='r', s=10)
plt.show()

# %% [markdown]
# How to add points of interest ? Have a look to the default parameters.
# %% [markdown]
# ### SIFT
# %% [markdown]
# Let's try the SIFT detector and descriptor. We will use the OpenCV implementation.

# %%
# if necessary installation :
#            !pip install opencv-contrib-python

print(cv2.__version__)


# %%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
cv2.drawKeypoints(
    gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img)


# %%
