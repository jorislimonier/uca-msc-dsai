library(ggplot2)
library(dplyr)

# Part 1
# Question 1a
movies <- read.csv("datasets_exam/movies.csv")

# Question 1b
head(movies)

# Question 1c
colnames(movies)

# Question 1d
dim(movies)
nrow(movies)
ncol(movies)

# Part 2
# Question 2a
movies %>% ggplot(aes(x=Tickets_Sold, y=Gross_Sales)) +
geom_point() # proportional relationship since tickets bring money

# Question 2b
cor(movies$Tickets_Sold, movies$Gross_Sales) # expected since tickets bring money

# Part 3
# Question 3a
movies %>% ggplot(aes(x=Genre, y=Tickets_Sold)) +
geom_boxplot(size=.5, color="black", fill="lightblue", rotate=45) +
scale_x_discrete(guide = guide_axis(angle = 45))

# Question 3b
movies %>% ggplot(aes(x=Genre, fill=Genre)) +
geom_bar() +
scale_x_discrete(guide = guide_axis(angle = 45)) +
NULL

# Question 3c & Question 3d
nbins <-25
movies %>% ggplot(aes(x=Tickets_Sold, fill=..x..)) +
geom_histogram(aes(y=..count..), bins=nbins, color="#555555") + # make slider for bins
scale_fill_gradient2(low='red', mid='red', high='green', midpoint=-.2*10^7, name="Tickets sold") +
stat_bin(aes(y=..count.., label=ifelse(..count..==0, "",..count..)), # display only nonzero bins
        geom="text", hjust=.6, vjust=-1, bins=nbins) +
scale_y_continuous(name="Count")
NULL


# Question 3e
movies %>% ggplot(aes(x=Genre, y=Tickets_Sold)) +
NULL