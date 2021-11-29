# install.packages("dplyr")
library(dplyr)
library(readxl)
chicago <- readRDS("chicago_data/chicago.rds")

dim(chicago)
str(chicago)
names(chicago)

# Take first 3 columns
names(chicago)[1:3]
subset <- select(chicago, city:dptp)
head(subset)

# Take everything except the first 3 columns
subset_minus <- select(chicago, -(city:dptp))
head(subset_minus)
