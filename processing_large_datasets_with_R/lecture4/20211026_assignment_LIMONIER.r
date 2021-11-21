library(dplyr)
library(ggplot2)
library(dslabs)
library(ggrepel)
library(gridExtra)

# Exercise 1
## Question 1 & Question 2
p1 <- murders %>% ggplot(aes(total, population, label=state)) +
geom_point(aes(color=region)) +
geom_text_repel() +
ggtitle("state")

p2 <- murders %>% ggplot(aes(total, population, label=abb)) +
geom_point(aes(color=region)) +
geom_text_repel() +
ggtitle("abb")

p3 <- murders %>% ggplot(aes(total, population, label=state)) +
geom_point(aes(color=region)) +
geom_text_repel() +
scale_x_log10() +
scale_y_log10() +
xlab("Population (log scale)") +
ylab("Total number of murders (log scale)") +
ggtitle("state log")

p4 <- murders %>% ggplot(aes(total, population, label=abb)) +
geom_point(aes(color=region)) +
geom_text_repel() +
scale_x_log10() +
scale_y_log10() +
xlab("Population (log scale)") +
ylab("Total number of murders (log scale)") +
ggtitle("abb log")

grid.arrange(p1, p2, p3, p4, ncol = 2, nrow=2)

# Exercise 2
## human characters
humans <- starwars %>%
filter(starwars$species == "Human")

## worlds
unique(starwars$homeworld)

## height and mass
### height
starwars %>%
group_by(species) %>%
mutate(mean_height_species=mean(height, na.rm=TRUE)) %>%
select(mean_height_species) %>%
unique()

### mass
starwars %>%
group_by(species) %>%
mutate(mean_mass_species=mean(mass, na.rm=TRUE)) %>%
select(mean_mass_species) %>%
unique()

## Plot number of characters of each type in decreasing order
species_count <- starwars %>%
count(species) # this keeps NA at the end

species_count %>%
ggplot(aes(reorder(species, -n), n)) +
geom_bar(stat="identity")

## Relationship between height and weight
starwars %>%
ggplot() +
geom_point(aes(height, mass)) # there is an outlier, but it seems that height and mass are positively correlated


# Exercise 3
dist1 <- rnorm(1000, 0, 1)
dist2 <- rnorm(1000, 5, 2)

hist(dist1)
hist(dist2)
