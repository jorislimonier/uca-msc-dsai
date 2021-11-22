# SECOND ASSIGNMENT
# -Part A)
# --Question 1
norm_vect <- rnorm(100, 0, 1)
# --Question 2
vect_mean <- 0
for (draw in norm_vect) {
    vect_mean <- vect_mean + draw
}
vect_mean <- vect_mean / length(norm_vect)

# --Question 3
variance <- 0
for (i in 1:100) {
    variance <- variance + (vect_mean - norm_vect[i])^2
}
variance <- variance / (length(norm_vect) - 1)
var(norm_vect)


# -Part B)
# --Question 1
ds <- airquality
library(dplyr)
for (colname in colnames(ds)){
    n_na <- sum(is.na(ds[colname]))
    col_len <- length(ds[[colname]])
    # --Question 2
    p_na <- n_na / col_len
    print(c(colname, p_na))

    # --Question 3
    if (p_na > .5) {
        ds <- select(ds, -colname)
    } else {
        # --Question 4
        ds[colname][is.na(ds[colname])] <- 0
    }
}


# -Part C)
for (colname in colnames(iris)[1:4]) {
    # --Question 1
    print("Mean")
    print(lapply(iris[colname], mean))
    # --Question 2
    print("Standard deviation")
    print(lapply(iris[colname], sd))
}


