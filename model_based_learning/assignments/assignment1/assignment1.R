# Tutorial EM for GMMs
#
# Your report must be a pdf, a jupyter notebook, or a R markdown.
#
# 1) Implementing the EM
#
# Implement (from scratch) the EM for a GMM on the variables 2 and 4 of the wine data set. Cluster the data and compare your results with k-means.
# An R file called "useful_functions.R" can be useful for EM. Apart from that, try not to use packages to implement EM.
# To assess the quality of the clustering, you may use the function classError and/or adjustedRandIndex from the Mclust package.
#
# 2) Model selection
#
# Try to find a relevant number of clusters using the three methods seen in class: AIC, BIC, and (cross-)validated likelihood.
#
# 3) Towards higher dimensional spaces
#
# Try to model more than just two variables of the same data set. Do you find the same clusters, the same number of clusters.
#


#### First we load the data and look at it

library(pgmm)
source("useful_functions.R")


data(wine)

X <- as.matrix(wine[, c(2, 4)])
y <- wine[, 1]
plot(X, col = y)
