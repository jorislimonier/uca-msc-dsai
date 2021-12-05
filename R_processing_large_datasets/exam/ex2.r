library(dplyr)


# EXERCISE 2

# Part 1
# Question 1a
winter <- read.csv("datasets_exam/winter_olympic.csv")

# Question 1b
head(winter)

# Question 1c
colnames(winter)

# Question 1d
dim(winter)
nrow(winter)
ncol(winter)

# Part 2
head(winter)
winter %>%
arrange(Total, NOC)

# Part 3
# Part 4
# Part 6
# Question 1a
# Question 1b
# Question 1c
# Question 1d
# Question 1e
