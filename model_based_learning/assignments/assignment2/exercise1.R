library(mice)

# The synthetic_mcar function takes as input:
#   - xna: a dataset containing missing values
#   - perc_na: the percentage of missing values we want to introduce in addition.
# This functions returns a list containing:
#   - xna_miss: a dataset containing the old and the new added missing values,
#   - mask: a vector with the indexes of the new added missing values
synthetic_mcar <- function(xna, perc_na) {
  true_na <- which(is.na(xna))
  nb_na <- round(sum(dim(xna)) * perc_na)
  synthetic_na <- sample(
    setdiff(seq_len(sum(dim(xna))), true_na),
    nb_na,
    replace = FALSE
  )
  xna_miss <- xna
  xna_miss[synthetic_na] <- NA
  return(list(xna_miss = xna_miss, mask = synthetic_na))
}

# Implement the mean imputation method
mean_imputation <- function(xna_miss) {
  xna_miss_mean <- xna_miss
  xna_miss_mean[is.na(xna_miss_mean)] <- mean(xna_miss_mean, na.rm = TRUE)
  return(xna_miss_mean)
}
# Implement the median imputation method
median_imputation <- function(xna_miss) {
  xna_miss_median <- xna_miss
  xna_miss_median[is.na(xna_miss_median)] <- median(xna_miss_median, na.rm = TRUE)
  return(xna_miss_median)
}
# Implement the multiple imputation method
multiple_imputation <- function(xna_miss) {
  xna_miss_mi <- mice(xna_miss, m = 5, maxit = 50, seed = 123, printFlag = FALSE)
  xna_miss_mi <- complete(xna_miss_mi, 1)
  return(xna_miss_mi)
}
