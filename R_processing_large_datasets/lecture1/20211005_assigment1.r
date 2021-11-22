# ASSIGNMENT 1
# 1A
sum_int <- function(n) {
    return(n*(n+1)/2)
}
sum_int(100)

# 1B
library(dslabs)
data(murders)
a<-murders$abb
class(a)

# 1C
b<-murders[, 2]
a == b

# 1D
table(murders)
table(murders$abb, murders$region)

# 1E
v<-c(1, 2, 3, 4, 5)
w<-c(42, 43, 44)
c(v[1:2], w, v[3:length(v)])

# 1F
spl<-runif(100, 0, 1)
sum(spl < .5)

# 1G
murders$murder_rate<-10**5 * murders$total / murders$population
mean_murder_rate<-mean(murders$murder_rate)

# 1H
# install.packages("readr")
library(readr)
vect<-data.frame(read_csv("vectors.csv", show_col_types=FALSE))
rem_from_vect <- function(vect, i){
    if (i == 1){
        low_lim <- 1
    } else {
        low_lim <- i-1
    }
    if (i == nrow(vect)){
        low_lim <- nrow(vect)
    } else {
        upp_lim <- i+1
    }
        c(vect[1:low_lim, 1], vect[upp_lim:nrow(vect), 1])
}
rem_from_vect(vect, 2)

# 1I
hist(murders$population)

# 1J
boxplot(population~region, data=murders)

# 1K
nmlz <- function(v){
    nm <- sum((abs(v)^2))^(1/2)
    return(v/nm)
}
v<-c(1, 2, 2)
nmlz(v)

# 1L
data(iris)
for (i in 1:(ncol(iris)-1)){
    iris[, i] <- nmlz(iris[, i])
}
