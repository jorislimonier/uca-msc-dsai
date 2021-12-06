library(dplyr)
library(ggplot2)

# Part 1

## Question 1a & Question 1b
swo <- read.csv("datasets_exam/summer_winter_olympics.csv")

dim(swo)
nrow(swo)
ncol(swo)
head(swo)

colnames(swo) <- c(
    "index",
    "NOC",
    "summer_played",
    "summer_gold",
    "summer_silver",
    "summer_bronze",
    "summer_total",
    "winter_played",
    "winter_gold",
    "winter_silver",
    "winter_bronze",
    "winter_total",
    "both_played",
    "both_gold",
    "both_silver",
    "both_bronze",
    "both_total"
)

## Question 1c
table(swo$summer_played)

## Question 1d
for (column in tail(colnames(swo), -2)) {
    print(column)
    print("FREQUENCY TABLE")
    print(table(swo[[column]]))
}
summary(swo)

# Part 4
# Question 4a

# Question 4b
# Question 4c
# Question 4d
# Question 4e
# Question 4f
# Question 4g
# Question 4h
# Question 4i