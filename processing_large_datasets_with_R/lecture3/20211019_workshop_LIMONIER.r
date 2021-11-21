# Exercise 1


# 1a
v <- seq(1, 10, .5)
v

# 1b
farenheit_to_celsius <- function (far) {
    return((far - 32)*5/9)
}
farenheit_to_celsius(-40)

# 1c
t <- sample(v, 100, replace=TRUE)
library(vctrs)
counts <- vec_count(t)
counts[1,1] # occurs the most
hist(t, main="Sampling with replacements", col="yellow")

# 1d
rand_mat <- matrix(0, 7, 8)
matrix(rnorm(rand_mat, 1, 2), 7, 8)

# 1e
library(dplyr)
ds <- iris[,-5]
boxplot(ds[1:length(ds)])

pairs(ds, col=as.numeric(iris$Species), pch=as.numeric(iris$Species))


# 1f (mistake: "shape", not "scale" in question)

plot(density(rgamma(1000, 1)))
lines(density(rgamma(1000, 2)))
lines(density(rgamma(1000, 3)))
lines(density(rgamma(1000, 4)))

# Exercise 2

# install.packages("nycflights13")
library(nycflights13)

# 2a
colnames(flights)
head(flights)
new_flights <- flights[which(flights$year==2013 & flights$month==4 & flights$day==8),]

min_delay <- min(new_flights$dep_delay[!is.na(new_flights$dep_delay)])
new_flights[which(new_flights$dep_delay==min_delay),] # flight with lowest departure delay

# 2b
colnames(flights)
four_hours <- 4 * 60
flights[which(flights$dep_delay>four_hours | flights$arr_delay>four_hours),]


most_delayed <- flights %>%
    group_by(month) %>% 
    mutate(mean_delay_month=mean(dep_delay, na.rm=TRUE)) %>% 
    filter(dep_delay == max(dep_delay, na.rm=TRUE)) %>% 
    select(dep_delay, month, day, mean_delay_month)

# 2c
lax <- flights %>% 
  filter(dest == "LAX") %>% 
  select(dep_delay, arr_delay) %>% 
  arrange(dep_delay)

mean(lax$dep_delay, na.rm=TRUE)
mean(lax$arr_delay, na.rm=TRUE)

# Exercise 3

# 3a
age <- runif(100)*20 + 20
weight <- round(runif(100)*40 + 50, 1)
grad <- rbinom(100, 1, .6)
typeof(lax)
people <- data.frame(age, weight, grad)

# 3b...there is no 3b

# 3c
for (col in colnames(people)){
    print(col)
    for (row in sample(seq(1, 100), 5)){
        people[row, col] <- NA
    }
}

# 3d
people
colnames(people)[3] <- "driving_license" # rename grad
sum(is.na(people)) # count NA
people <- people %>% filter(!is.na(weight), !is.na(age), !is.na(driving_license))

# 3e
people$weight <- (people$weight - min(people$weight)) / (max(people$weight) - min(people$weight))
people$age <- (people$age - min(people$age)) / (max(people$age) - min(people$age))


# 3f
people$weight <- (people$weight - mean(people$weight)) / sd(people$weight)
people$age <- (people$age - mean(people$age)) / sd(people$age)

