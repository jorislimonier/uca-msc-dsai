# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ### Functions
# 
# In Python, we can define a function by using keyword def.

# %%
def square(x):
    return x*x

print(square(5))

# %% [markdown]
# You can apply a function to each element of a list/array by using lambda function. For example, we want to square elements in a list:

# %%
array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# apply function "square" on each element of "array"
print(list(map(lambda x: square(x), array)))

# or using a for loop, and a list comprehension
print([square(x) for x in array])

print("orignal array:", array)

# %% [markdown]
# These two above syntaxes are used very often.
# 
# If you are not familiar with list comprehensions, follow this [link](http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Comprehensions.html%5D).
# 
# We can also put a function B inside a function A (that is, we can have nested functions). In that case, function B is only accessed inside function A (the scope that it's declared). For example:

# %%
# select only the prime number in array
# and square them
def filterAndSquarePrime(arr):
    
    # a very simple function to check a number is prime or not
    def checkPrime(number):
        if (number <= 1): return False
        # if we find any divisor of "number" return false
        # we can improve this process by only finding divisors in range [2, sqrt(number)]
        for i in range(2, int(number/2) + 1):
            if number % i == 0:
                return False
        return True
    
    primeNumbers = filter(lambda x: checkPrime(x), arr)
    return map(lambda x: square(x), primeNumbers)

# we can not access checkPrime from here
# checkPrime(5)

result = filterAndSquarePrime(array)
list(result)

# %% [markdown]
# ### Importing modules, functions
# 
# Modules in Python are packages of code. Putting code into modules helps increasing the reusability and maintainability. The modules can be nested. To import a module, we simple use syntax: import <module_name>. Once it is imported, we can use any functions, classes inside it.

# %%
# import module 'math' to uses functions for calculating
import math

# print the square root of 16
print(math.sqrt(16))

# we can create alias when import a module
import numpy as np

print(np.sqrt(16))

# %% [markdown]
# Sometimes, you only need to import some functions inside a module to avoid loading the whole module into memory. To do that, we can use syntax: from <module> import <function>

# %%
# only import function 'sin' in package 'math'
from math import sin

# use the function
print(sin(60))

# %% [markdown]
# ## ===> Your turn!
# %% [markdown]
# ## Question 1
# %% [markdown]
# ### 1.1
# 
# Write a function `checkSquareNumber` to check if a integer number is a square number or not. For example, 16 and 9 are square numbers. 15 isn't square number.
# Requirements: - **Input**: an integer number - **Output**: `True` or `False`
# 
# *hint: if the square root of number x is an integer, then x is a square number*

# %%
import math

def checkSquareNumber(x):
    ...
    
print(checkSquareNumber(16))
print(checkSquareNumber(250))

# %% [markdown]
# ### 1.2
# 
# A list list_numbers which contains the numbers from 1 to 9999 can be constructed from:
# 
# `list_numbers = range(0, 10000)`
# 
# Extract the square numbers in `list_numbers` using function `checkSquareNumber` from question 1.1. How many elements in the extracted list ?

# %%
list_numbers = range(0, 10000)
square_numbers = ...
print(square_numbers)
print(len(square_numbers))

# %% [markdown]
# ### 1.3
# 
# Using array slicing, select the elements of the list square_numbers, whose index is from 5 to 20 (zero-based index).

# %%
print(...)

# %% [markdown]
# ## 2.2 Numpy
# 
# Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.
# 
# ### 2.2.1. Array
# A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. Thanks to the same type property, Numpy has the benefits of locality of reference. Besides, many other Numpy operations are implemented in C, avoiding the general cost of loops in Python, pointer indirection and per-element dynamic type checking. So, the speed of Numpy is often faster than using built-in datastructure of Python. When working with massive data with computationally expensive tasks, you should consider to use Numpy.
# 
# The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.
# 
# We can initialize numpy arrays from nested Python lists, and access elements using square brackets:

# %%
import numpy as np

# Create a rank 1 array
rank1_array = np.array([1, 2, 3])
print("type of rank1_array:", type(rank1_array))
print("shape of rank1_array:", rank1_array.shape)
print("elements in rank1_array:", rank1_array[0], rank1_array[1], rank1_array[2])

# Create a rank 2 array
rank2_array = np.array([[1,2,3],[4,5,6]])
print("shape of rank2_array:", rank2_array.shape)
print(rank2_array[0, 0], rank2_array[0, 1], rank2_array[1, 0])

# %% [markdown]
# ### 2.2.2. Array slicing
# Similar to Python lists, numpy arrays can be sliced. The different thing is that you must specify a slice for each dimension of the array because arrays may be multidimensional.

# %%
import numpy as np
m_array = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

print(m_array[[0,1]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2
b = m_array[:2, 1:3]
print(b)

# we can only use this syntax with numpy array, not python list
print("value at row 0, column 1:", m_array[0, 1])

# Rank 1 view of the second row of m_array  
print("the second row of m_array:", m_array[1, :])

# print element at position (0,2) and (1,3)
print(m_array[[0,1], [2,3]])

# %% [markdown]
# ### 2.2.3. Boolean array indexing
# We can use boolean array indexing to check whether each element in the array satisfies a condition or use it to do filtering.

# %%
m_array = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Find the elements of a that are bigger than 2
# this returns a numpy array of Booleans of the same
# shape as m_array, where each value of bool_idx tells
# whether that element of a is > 3 or not
bool_idx = (m_array > 3)
print(bool_idx , "\n")

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(m_array[bool_idx], "\n")

# We can combine two statements
print(m_array[m_array > 3], "\n")

# select elements with multiple conditions
print(m_array[(m_array > 3) & (m_array % 2 == 0)])

# %% [markdown]
# ### 2.2.4. Datatypes
# Remember that the elements in a numpy array have the same type. When constructing arrays, Numpy tries to guess a datatype when you create an array However, we can specify the datatype explicitly via an optional argument.

# %%
# let Numpy guess the datatype
x1 = np.array([1, 2])
print(x1.dtype)

# force the datatype be float64
x2 = np.array([1, 2], dtype=np.float64)
print(x2.dtype)

# %% [markdown]
# ### 2.2.5. Array math
# Similar to Matlab or R, in Numpy, basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module.

# %%
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
# mathematical function is used as operator
print("x + y =", x + y, "\n")

# mathematical function is used as function
print("np.add(x, y)=", np.add(x, y), "\n")

# Unlike MATLAB, * is elementwise multiplication
# not matrix multiplication
print("x * y =", x * y , "\n")
print("np.multiply(x, y)=", np.multiply(x, y), "\n")
print("x*2=", x*2, "\n")

# to multiply two matrices, we use dot function
print("x.dot(y)=", x.dot(y), "\n")
print("np.dot(x, y)=", np.dot(x, y), "\n")

# Elementwise square root
print("np.sqrt(x)=", np.sqrt(x), "\n")

# %% [markdown]
# Note that unlike MATLAB, `*` is elementwise multiplication, not matrix multiplication. We instead use the `dot` function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices. In what follows, we work on a few more examples to reiterate the concept.

# %%
# declare two vectors
v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors
print("v.dot(w)=", v.dot(w))
print("np.dot(v, w)=", np.dot(v, w))

# Matrix / vector product
print("x.dot(v)=", x.dot(v))
print("np.dot(x, v)=", np.dot(x, v))

# Matrix / matrix product
print("x.dot(y)=", x.dot(y))
print("np.dot(x, y)=", np.dot(x, y))

# %% [markdown]
# Additionally, we can do other aggregation computations on arrays such as `sum`, `nansum`, or `T`.

# %%
x = np.array([[1,2], [3,4]])

# Compute sum of all elements
print(np.sum(x))

# Compute sum of each column
print(np.sum(x, axis=0))

# Compute sum of each row
print(np.sum(x, axis=1))

# transpose the matrix
print(x.T)

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v.T)  # Prints "[1 2 3]"

# %% [markdown]
# ## ===> Your turn!
# %% [markdown]
# ## Question 2
# 
# Given a 2D array:
# 
# ```
#  1  2  3  4
#  5  6  7  8 
#  9 10 11 12
# 13 14 15 16
# ```
# %% [markdown]
# ### 2.1
# 
# Print the all odd numbers in this array using Boolean array indexing.

# %%
import numpy as np
array_numbers = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

print(...)

# %% [markdown]
# ### 2.2
# 
# Extract the second row and the third column in this array using array slicing.

# %%
print(...)
print(...)

# %% [markdown]
# ### 2.3
# Calculate the sum of diagonal elements.

# %%
sum_ = 0


print(sum_)

# %% [markdown]
# ### 2.4
# Print elementwise multiplication of the first row and the last row using numpy's functions. Print the inner product of these two rows.

# %%
print(...)
print(...)

print(...)

# %% [markdown]
# ## 2.3 Matplotlib
# 
# As its name indicates, Matplotlib is a plotting library. It provides both a very quick way to visualize data from Python and publication-quality figures in many formats. The most important function in matplotlib is plot, which allows you to plot 2D data.

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('custom y label')
plt.show()

# %% [markdown]
# In this case, we provide a single list or array to the plot() command, matplotlib assumes it is a sequence of y values, and automatically generates the x values for us. Since python ranges start with 0, the default x vector has the same length as y but starts with 0. Hence the x data are [0,1,2,3].
# 
# In the next example, we plot figure with both x and y data. Besides, we want to draw dashed lines instead of the solid in default.

# %%
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r--')
plt.show()

plt.bar([1, 2, 3, 4], [1, 4, 9, 16], align='center')
# labels of each column bar
x_labels = ["Type 1", "Type 2", "Type 3", "Type 4"]
# assign labels to the plot
plt.xticks([1, 2, 3, 4], x_labels)

plt.show()

# %% [markdown]
# If we want to merge two figures into a single one, subplot is the best way to do that. For example, we want to put two figures in a stack vertically, we should define a grid of plots with 2 rows and 1 column. Then, in each row, a single figure is plotted.

# %%
# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r--')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.bar([1, 2, 3, 4], [1, 4, 9, 16])

plt.show()

# %% [markdown]
# For more examples, please visit the [homepage](http://matplotlib.org/1.5.1/examples/index.html) of Matplotlib.
# %% [markdown]
# ## ==> Your turn
# %% [markdown]
# ## Question 3
# 
# Given a list of numbers from 0 to 9999.
# 
# ### Question 3.1
# 
# Calculate the histogram of numbers divisible by 3, 7, 11 in the list respectively.
# ( Or in other word, how many numbers divisible by 3, 7, 11 in the list respectively ?
# 

# %%
def divisor(x, y):
    return x % y == 0


arr = np.array(range(0,10000))
divisors = ...
histogram = ...
print(histogram)

# %% [markdown]
# ### Question 3.2
# Plot the histogram in a line chart.

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# %% [markdown]
# ### Question 3.3
# Plot the histogram in a bar chart.

# %%



