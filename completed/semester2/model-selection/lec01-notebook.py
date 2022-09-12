# %% [markdown]
# # Basic Probability Models and Sampling in Python
# 
# This lesson introduces the basic concepts of sampling and computing with probability models in Python. 
# 
# Numpy provides a rich variety of functions and models that will help us in developing our computing tools.
# 
# Let's start by generating samples from an univariate Gaussian distribution with given mean $\mu$ and standard deviation $\sigma$.

# %%
import numpy as np

mu = 100
sigma = 20
samples = 300

data = np.random.normal(mu, sigma, samples)

# %% [markdown]
# Having generated the sample, we can inspect the properties of the data to get more information about our distribution.
# 
# The first question we would like to ask is what is the ratio of our data being greather than a certain value $L$. This can be approximated by counting: 

# %%
L = 120
prob = float(np.sum(data>120))/samples

print("The ratio of values being greater than " + str(L) + " is: p(x>"+str(L)+") = " + str(prob))

# %% [markdown]
# Similarly, we can compute the ratio of samples $L_1<x<L_2$:

# %%
L1 = 100
L2 = 120

prob = float(np.sum((data>100) & (data<120)))/samples

print("The ratio is: p("+str(L1)+"<x<"+str(L2)+") = " + str(prob))

# %% [markdown]
# Summary statistics can be simply computed as:

# %%
mean = np.sum(data)/samples
std = np.sqrt(np.sum((data-mean)**2)/samples)

print("Sample mean: " + str(mean) + "\nSample std: " + str(std))

print("\nNumpy functions:")
print(f"mean: {np.mean(data)}, std: {np.std(data)}")


# %% [markdown]
# $\mathbf{Exercise:}$ Compute the median of the sample.

# %%
import plotly.express as px
import plotly.graph_objects as go
median = np.sort(data)[np.floor(samples/2).astype(int) -
                       1: np.ceil(samples/2).astype(int)+1].mean()
px.histogram({"sampled data": data}).add_vline(
    x=median, line_color="red").show()
median, np.median(data)


# %% [markdown]
# An important way of visualizing a distribution is through an histogram plot. A histogram plot is created by discretizing the domain of the distribution in a certain number of bins, and by computing the number of realization of the distribution falling within each bin. 

# %%
first_edge, last_edge = data.min(), data.max()
n_equal_bins = 10
bin_edges = np.linspace(start=first_edge, stop=last_edge,
                        num=n_equal_bins + 1, endpoint=True)

# All but the last (righthand-most) bin is half-open.
bin_height = []
for i in range(len(bin_edges)-2):
    bin_height.append(np.sum((data >= bin_edges[i]) & (data < bin_edges[i+1])))

# The last bin is closed on the right
bin_height.append(np.sum((data >= bin_edges[i+1]) & (data <= bin_edges[i+2])))

bin_height = np.array(bin_height)
print('\n Histogram computation')
print(bin_edges, bin_height)

# Handmade histogram
print('\n Handmade histogram plot:\n')

bin_centers = np.diff(bin_edges)/2 + bin_edges[:-1]

for i in range(n_equal_bins):
    print(f"{round(bin_centers[i], 3).astype(str).zfill(7)} {'+' * bin_height[i]}")

# The same histogram can be obtained with numpy
np_bin_height, np_bin_edges = np.histogram(data)

print('\n Numpy default histogram')
print(np_bin_edges, np_bin_height)


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)

n, bins, patches = plt.hist(x=data, bins=10, color='#0504aa',
                            alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Matplotlib histogram')
plt.text(140, 45, r'$mu=' + str(mu) + ', \nsigma= ' + str(sigma) +' $')
maxfreq = n.max()

# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.subplot(1, 2, 2)
n, bins, patches = plt.hist(x=data, density = True, bins=10, color='#0504aa',
                            alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Matplotlib normalized histogram')
plt.text(140, 0.006, r'mu=' + str(mu) + ', \nsigma= ' + str(sigma) +' $')
maxfreq = n.max()
plt.ylim(ymax=maxfreq + 0.0005)

bin_size = bins[1]-bins[0]

print("The sum of the bin heights in the normalized histogram is: " + str(np.sum(n*bin_size)))

# %% [markdown]
# The probability density function (PDF) $f(x)$ of a random variable is a function quantifying the density of the variable at each point $x$ of the domain. By integrating the PDF, $\int_{a}^b f(t)\, dt$, we quantify the probability of the random variable falling within a particular range of values. 
# In our case, when dealing with discrete samples of a variable, a discrete approximation of the PDF is provided by the normalization histogram. The PDF is equivalent to a continuous representation of the histogram, and can be estimated through interpolation.

# %%
from scipy import stats

fine_range = np.linspace(10,180,200)

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
interpolated = np.interp(fine_range,bins[1:],n)
plt.plot(fine_range, interpolated)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('PDF through linear interpolation')

plt.subplot(1, 2, 2)
gkde = stats.gaussian_kde(dataset = data)
plt.plot(fine_range, gkde.evaluate(fine_range))
plt.title('PDF through kernel density estimator')

plt.show()

# %% [markdown]
# $\mathbf{Exercise:}$ Plot the cumulative density function (CDF) of the sample. The CDF of $f(x)$ is defined as $F(x) = \int_{-\infty}^x f(t)\, dt$. In our discrete setting, the integral will be approximated by a sum.

# %%
data = np.sort(data)
px.scatter(np.cumsum(data) / np.sum(data))
tickers = np.linspace(np.min(data), np.max(data), num=500)
emp_cdf = np.array(
    [np.sum(data < tickers[i]) for i in range(len(tickers))]
) / samples
px.line(emp_cdf).show()
np.sum(emp_cdf)


# %%
px.line(np.cumsum(gkde.evaluate(fine_range)))

# %% [markdown]
# ## Sampling in Python
# 
# As we have seen, sampling is a fundamental operation at the basis of a great number of procedures in computer science.
# In this section we will discover useful functionalities that will allow us to control sampling operations on numerical objects. 
# 
# The first important point is the concept of $seed$: a computer program can generate does provide pseudo-random numbers, that are generated starting from an initial value called, indeed, seed. Let's see how this work in practice:
# 

# %%
# Setting the seed
np.random.seed(123)

# Computing random numbers
print("Ten numbers sampled from the seed 123: \n" + str(np.random.random(10)))

# Changing the seed
np.random.seed(321)
print("Ten numbers sampled from the seed 321: \n" + str(np.random.random(10)))

#Back to the initial seed
np.random.seed(123)
print("Ten numbers sampled again from the seed 123: \n" + str(np.random.random(10)))

# %% [markdown]
# The concept of seed is very important for reproducibility across different experiments. Every time we set the same seed we ensure to reproduce the same set of (pseudo-)random numbers.
# 
# In addition to sampling from a theoretical distribution, such as normal or uniform, we can sample from a given array of values. This operation sets the basis of the resampling methods that we will see in the future lessons.

# %%
np.random.seed(10)

int_array = np.random.randint(0,50,20)

print("Here is a randomly generated sequence of 20 integers: \n" + str(int_array))

selected = np.random.choice(int_array,5)

print("Here we selected only 5 elements of the sequence: \n" + str(selected))

# %% [markdown]
# We notice however that  the number 8 was selected twice. If we want to ensure that each element is sampled only once, we need to set the option `replace=False`.

# %%
selected = np.random.choice(int_array,5,replace=False)

print("Here we selected 5 elements without replacement: \n" + str(selected))

# %% [markdown]
# The same procedure can be applied for sampling for more complex objects, such as multidimensional arrays (tensors):

# %%
array2d = np.random.randint(0, 10, (8, 12))
print(array2d)

# %%
print(array2d.shape[0])
print(np.random.choice(8, 4, replace=False))

line_count = np.arange(8)
print(line_count)

print(np.random.choice(line_count, 4, replace=False))

# %%
#Here we sample 4 rows
idx = np.random.choice(array2d.shape[0], 4)
print("Four randomly sampled rows in array2d: \n" + str(array2d[idx, :]))

#Here we sample 4 columns
idx = np.random.choice(array2d.shape[1], 4)
print("Four randomly sampled columns in array2d: \n" + str(array2d[:, idx]))


# %% [markdown]
# We can also randomly reshuffle the elements of an array. Be careful, the command `np.random.shuffle` reshuffles the elements along the first axis only. By default, the reshuffling is performed in place:

# %%
print(array2d)

np.random.shuffle(array2d)

print("Reshuffled array2d across rows: \n" + str(array2d))


# %% [markdown]
# $\mathbf{Exercise.}$ How would you reshuffle array2d across columns?

# %%
array_transp = array2d.T
np.random.shuffle(array_transp)
array_transp.T

# %% [markdown]
# 
# $\mathbf{Exercise.}$ Write your own reshuffling routine.

# %%
n = 5
# np.random.seed(3)
arr = np.random.randint(0, 20, 5)
indices = np.random.choice(a=np.arange(n), size=n, replace=False)
arr, arr[indices]

# %% [markdown]
# ## In-depth: how random variables can be generated by a computer program
# 
# $\textit{Anyone who considers arithmetical methods of producing random digits is, of course, in a state of sin.}$ (John von Neumann)
# 
# Generation or simulation of random numbers, using deterministic algorithms, is widely used by statisticians for several purposes. A large number of (pseudo-)random number generators have been studied in the past, starting from very long time ago:
# 
# ``Brother Edvin (a monk), sometime between 1240 and 1250 AD, was preparing a case for the sainthood of King Olaf Haraldsson, who had been the King of Norway. There was a well documented story (that could still be false) that King Olaf and the King of Sweden needed to determine which country owned the Island the Hising. They agreed to determine this by chance. They were using normal 6-sided dice. The King of Sweden rolled two dice and got a 12. Then King Olaf rolled and got a 12 AND one of the dice broke (hence the name of the book) and he got an additional 1 for a 13. Some attributed this event to divine intervention, which strengthened his case for sainthood.’’ (https://blog.computationalcomplexity.org)
# 
# According to the story, this even motivated Brother Edvin in thinking of an algorithm to generate random numbers in a way that nobody can manipulate them. A very similar algorithm was reproposed by von Neumann in 1946 for creating the so-called middle-square method.  
# 
# Many mehtods have been proposed since them, and an overview is for example provided in wikipedia:  (https://en.wikipedia.org/wiki/List_of_random_number_generators).
# 
# In this section we will assume that we can indeed generate random numbers uniformly distributed in the interval $[0,1]$. This actually corresponds to sampling from the uniform distribution $\mathbf{U}(0,1)$. 
# 
# 
# For the sake of illustration, we will focus on classical examples on the generation of samples from standard distributions, such as Exponential and Gaussian ones. In particular, in what follows we are going to illustrate some classical simulation techniques and of the underlying rationale.

# %% [markdown]
# ## Exponential 
# 
# The key property that can be used to sample from the exponential distribution is that the cumulative density function (CDF) $F(X)$ of a random variable $X$ is uniformely distributed $[0,1]$. Therefore, by applying the inverse CDF $F^{-1}$ to our uniform samples we can obtain data with the same distribution of $X$.
# 
# 
# For the exponential distribution, we can use this property to lead to a very simple generator. Since the probability density function of the exponential distribution is:
# 
# $$f(x) = \lambda\exp(-\lambda x)$$
# 
# it follows that the CDF of the exponential with parameter $\lambda$ is:
# 
# $$F(x) =  1-\exp(-\lambda x).$$
# 
# The inverse of $F$ can be easily derived:
# 
# $$F^{-1}(p) = -\frac{\log(1 - p)}{\lambda}.$$
# 
# Let's verify this empirically:

# %%
l = 1

unif_sample = np.random.uniform(0,1,1000)
exp_sample = -np.log(1-unif_sample)/l
plt.hist(exp_sample, histtype='step', bins=100, density=True, linewidth=2, label = 'random generated data')

def exp_pdf(x,l):
    return l*np.exp(-x*l)

x = np.linspace(0,10)
plt.plot(x, exp_pdf(x,l), linewidth=2, label = 'Exponential PDF')
plt.legend()
plt.title('Generating Exponentially distributed samples')
plt.show()




# %% [markdown]
# ## Gaussian
# 
# 

# %% [markdown]
# For the Gaussian case, sampling through the coomputation of the inverse CDF is more complicated, as the inverse of the Gaussian CDF cannot be written in closed form. We can however rely on the several available approximations, such as the one provided by H. Shore in 1982:
# 
# $$F^{-1}(p) \simeq 5.5556\left[ 1- \left(\frac{1-p}{p}\right)^{0.1186}\right], \qquad p\geq0.5$$
# $$F^{-1}(p) \simeq -5.5556\left[ 1- \left(\frac{p}{1-p}\right)^{0.1186}\right], \qquad p<0.5$$
# 
# 
# 
# 
# [Haim Shore. Simple Approximations for the Inverse Cumulative Function, the Density Function and the Loss Integral of the Normal Distribution. Journal of the Royal Statistical Society. Series C (Applied Statistics)
# Vol. 31, No. 2 (1982), pp. 108-114] 

# %%
unif_sample = np.random.uniform(0.5,1,1000)

def inv_norm_cdf(x):
    a = x[:int(len(x)/2)]
    b = x[int(len(x)/2):]
    z_top = 5.5556 * (1 - np.power((1-a)/a,0.1186))
    z_bottom = -5.5556 * (1 - np.power((1-b)/b,0.1186))
    return np.hstack([z_top,z_bottom])


normal_sample = inv_norm_cdf(unif_sample)

plt.hist(normal_sample, histtype='step', bins=100, density=True, linewidth=2, label = 'random generated data')


def normal_pdf(x):    
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

x = np.linspace(-5,5)
plt.plot(x, normal_pdf(x), linewidth=2, label = 'Normal PDF')
plt.legend()
plt.title('Generating Normal distributed samples')

plt.show()


# %% [markdown]
# Sampling can be also obtained by leveraging on other properties of probability models. For example, the central limit theorem states that, for independent and identically distributed random variables $X_1, \ldots, X_n$ with mean $\mu$ and variance $\sigma^2$, their average $S_n = \frac{X_1 + \ldots + X_n}{n}$ converges in distribution to the Gaussian according to the following relationhsip:  
# 
# $$ \sqrt{n}(S_n - \mu) \rightarrow \mathcal{N}(0,\sigma^2)$$
# 
# We can use this relationship to generate standard Gaussian sample from a independent samples from a uniform distribution $\mathbf{U}(0,1)$. Let $X_1, \ldots, X_{12}$ be 12 iid samples from $\mathbf{U}(0,1)$, and $S_{12} = \frac{X_1 + \ldots + X_{12}}{12}$. We know that the mean of $\mathbf{U}(0,1)$ is $\mu = \frac{1}{2}$, while the variance is $\frac{1}{12}$. Therefore, the above relationship writes as:
# 
# $$ \sqrt{12}(S_{12} - \frac{1}{2}) \rightarrow \mathcal{N}(0,\frac{1}{12})$$
# 
# Thanks to the multiplicative property of the variance of the Gaussian distribution, by multiplying both sides by $\sqrt{12}$ we obtain:
# 
# $$ (12 \cdot S_{12} - 6) \rightarrow \mathcal{N}(0,1) \\
# (X_1 + \ldots + X_{12} - 6) \rightarrow \mathcal{N}(0,1) $$
# 
# This gives us an interesting way to generate Gaussian distributed data from 12 Uniform samples.
# 

# %%
unif_sample = [ np.random.uniform(0,1,1000) for i in range(12)]

print(len(unif_sample))
print(len(unif_sample[0]))

# %%
unif_sample = [ np.random.uniform(0,1,1000) for i in range(12)]

normal_sample2 = (np.sum(unif_sample,0) - 6)

plt.hist(normal_sample2, histtype='step', bins=100, density=True, linewidth=2, label = 'random generated data')


plt.plot(x, normal_pdf(x), linewidth=2, label = 'Normal PDF')

plt.legend()
plt.title('Generating normally distributed samples')

plt.show()


# %% [markdown]
# A last general approach consists in the so-called Rejection Sampling procedure. We assume that the PDF $f(x)$ can be computed in closed form. The procedure is as follows:
# 
# - we identify the max of the PDF: $f_{max}$,
# - for each data point $x$ we generate random samples uniformly distributed $\mathbf{U}(0,f_{max})$, 
# - we then record the number of random samples with value lower than $f(x)$, the actual PDF at x.
# 
# We end up with an histogram associating to each point $x$ the number of elements with relative likelihood approximating $f(x)$.
# 

# %%
u = np.random.uniform(-5, 5, 1000)

r = np.random.uniform(0, normal_pdf(0), 1000)

v = u[r < normal_pdf(u)]

plt.hist(v, histtype='step', bins=100, density=True,
         linewidth=2, label='random generated data')
plt.plot(x, normal_pdf(x), linewidth=2, label='Normal PDF')

plt.legend()
plt.title('Generating Normal distributed samples')

plt.show()


# %%
u = np.random.uniform(0, 10, 10000)
r = np.random.uniform(0, l, 10000)

v = u[r < exp_pdf(u, l)]

plt.hist(v, histtype='step', bins=100, density=True,
         linewidth=2, label='random generated data')
x = np.linspace(0, 10)
plt.plot(x, exp_pdf(x, l), linewidth=2, label='Exponential PDF')


plt.show()


# %%


# %%



