library(dplyr)

# Question 1a
hist(iris$Sepal.Length)

# Question 1b
library(ggplot2)
iris %>% ggplot(aes(Sepal.Length)) +
geom_histogram(bins=50, fill="blue")

# Question 1c & Question 1d & Question 1e
iris %>% ggplot(aes(Sepal.Length, Sepal.Width, color=Species, shape=Species)) +
geom_point() +
geom_smooth(method="lm", se=FALSE) +
geom_smooth(method='loess')

# Question 1f
iris %>% ggplot(aes(Sepal.Length, Sepal.Width, color=Species, shape=Species)) +
geom_point() +
geom_smooth(method="lm", se=FALSE) +
facet_wrap(~Species)


# Question 2a
mpg %>% ggplot(aes(displ, hwy)) +
geom_point()

# Question 2b
mpg %>% ggplot(aes(displ, hwy, color=class, shape=as.factor(year))) +
geom_point()

# Question 2c
mpg %>% 
ggplot(aes(displ, hwy, color=class, shape=as.factor(year))) +
geom_point() +
facet_wrap(~class)

# Question 2d
diamonds %>%
ggplot(aes(carat, price)) +
geom_point(mapping=aes(col=cut)) +
geom_smooth(method="lm", se=FALSE) +
facet_wrap(~color)



# Question 3a
which.max(colSums(is.na(starwars))) # column birth year

# Question 3b
starwars %>%
filter(species=="Human") %>%
group_by(gender) %>%
count()

# Question 3c
starwars %>%
group_by(homeworld) %>%
count() %>%
arrange(desc(n))

# Question 3d
colors <- c("pink", "blue")

starwars %>%
ggplot(aes(gender)) +
geom_bar(aes(fill=gender)) +
scale_fill_manual(values=colors, na.value="black")
ggtitle("Gender distribution of the sw Universe")


# Question 3f
starwars %>%
group_by(gender) %>%
na.omit() %>%
ggplot(aes(height, fill=gender)) +
geom_density()

# Question 3g
starwars %>%
ggplot(aes(sex, fill=hair_color)) +
geom_bar(position="fill") +
labs(y="proportion")




