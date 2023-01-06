### The synthetic_mcar function takes as input
# xna: a dataset containing missing values
# perc_na: the percentage of missing values we want to introduce in addition.
### This functions returns a list containing
# xna_miss: a dataset containing the old and the new added missing values,
# mask: a vector with the indexes of the new added missing values

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
