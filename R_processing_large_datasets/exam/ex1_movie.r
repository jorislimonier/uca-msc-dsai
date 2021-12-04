library(dplyr)
library(ggplot2)
library(gridExtra)

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
scat_ticket_gross <- movies %>% ggplot(aes(x=Tickets_Sold, y=Gross_Sales)) +
geom_point() # proportional relationship since tickets bring money

# Question 2b
corr_scatter <- cor(movies$Tickets_Sold, movies$Gross_Sales) # expected since tickets bring money

# Part 3
# Question 3a
boxplot_all <- movies %>% ggplot(aes(x=Genre, y=Tickets_Sold)) +
geom_boxplot(size=.5, color="black", fill="lightblue") +
scale_x_discrete(guide = guide_axis(angle = 45))

# Question 3b
hist_genre <- movies %>% ggplot(aes(x=Genre, fill=Genre)) +
geom_bar() +
scale_x_discrete(guide = guide_axis(angle = 45)) +
NULL

# Question 3c & Question 3d

nbins <-25
hist_ticket_sales <- function(nbins) {
        movies %>% ggplot(aes(x=Tickets_Sold, fill=..x..)) +
        geom_histogram(aes(y=..count..), bins=nbins, color="#555555") + # make slider for bins
        scale_fill_gradient2(low='red', mid='red', high='green', midpoint=-.2*10^7, name="Tickets sold") +
        stat_bin(aes(y=..count.., label=ifelse(..count..==0, "",..count..)), # display only nonzero bins
                geom="text", hjust=.6, vjust=-1, bins=nbins) +
        scale_y_continuous(name="Count")
}

# Question 3e --> waiting for answer from Silvia BOTTINI
# movies %>% ggplot(aes(x=Genre, y=Tickets_Sold)) +
# NULL

# Exercise 4
# Question 4a - average number of tickets sold
av_ticket_genre <- function() {
        return (movies %>%
        group_by(Genre) %>%
        summarise(average_tickets_sold=mean(Tickets_Sold, na.rm = TRUE)))
}
av_ticket_genre()

# Question 4b - average gross sales
av_sales_genre <- function() {
        return (movies %>%
        group_by(Genre) %>%
        summarise(average_gross_sales=mean(Gross_Sales, na.rm = TRUE)))
}
av_sales_genre()

# Exercise 5
av_ticket_distr <- function() {
        return (movies %>%
        group_by(Distributor) %>%
        summarise(average_tickets_sold=mean(Tickets_Sold, na.rm = TRUE)))
}
av_ticket_distr()

av_sales_distr <- function() {
        return (movies %>%
        group_by(Distributor) %>%
        summarise(average_gross_sales=mean(Gross_Sales, na.rm = TRUE)))
}
av_sales_distr()

av_ticket_both <- function() {
        return (movies %>%
        group_by(Genre, Distributor) %>%
        summarise(average_tickets_sold=mean(Tickets_Sold, na.rm = TRUE)))
}
av_ticket_both()

av_sales_both <- function() {
        return (movies %>%
        group_by(Genre, Distributor) %>%
        summarise(average_gross_sales=mean(Gross_Sales, na.rm = TRUE)))
}
av_sales_both()

av_ticket <- function(by_genre, by_distr) {
        if (by_genre & by_distr) {
                return(av_ticket_both())
        } else if (by_genre) {
                return(av_ticket_genre())
        } else if (by_distr) {
                return(av_ticket_distr())
        } else{
                print("Nothing is computed.")
        }
}
av_ticket(by_genre=TRUE, by_distr=TRUE)
av_ticket(by_genre=FALSE, by_distr=TRUE)
av_ticket(by_genre=TRUE, by_distr=FALSE)
av_ticket(by_genre=FALSE, by_distr=FALSE)

av_sales <- function(by_genre, by_distr) {
        if (by_genre & by_distr) {
                return(av_sales_both())
        } else if (by_genre) {
                return(av_sales_genre())
        } else if (by_distr) {
                return(av_sales_distr())
        } else{
                print("Nothing is computed.")
        }
}
av_sales(by_genre=TRUE, by_distr=TRUE)
av_sales(by_genre=FALSE, by_distr=TRUE)
av_sales(by_genre=TRUE, by_distr=FALSE)
av_sales(by_genre=FALSE, by_distr=FALSE)

av_metric <- function(by_genre, by_distr, metric) {
        if (metric == "ticket") {
                av_ticket(by_genre=by_genre, by_distr=by_distr)
        } else if (metric == "sales") {
                av_sales(by_genre=by_genre, by_distr=by_distr)
        } else {
                print("Unknown metric")
        }
}
# test `ticket` metric
av_metric(by_genre=TRUE, by_distr=TRUE, metric="ticket")
av_metric(by_genre=FALSE, by_distr=TRUE, metric="ticket")
av_metric(by_genre=TRUE, by_distr=FALSE, metric="ticket")
av_metric(by_genre=FALSE, by_distr=FALSE, metric="ticket")

# test `sales` metric
av_metric(by_genre=TRUE, by_distr=TRUE, metric="sales")
av_metric(by_genre=FALSE, by_distr=TRUE, metric="sales")
av_metric(by_genre=TRUE, by_distr=FALSE, metric="sales")
av_metric(by_genre=FALSE, by_distr=FALSE, metric="sales")

# test `NONSENSE` metric
av_metric(by_genre=TRUE, by_distr=TRUE, metric="NONSENSE")
av_metric(by_genre=FALSE, by_distr=TRUE, metric="NONSENSE")
av_metric(by_genre=TRUE, by_distr=FALSE, metric="NONSENSE")
av_metric(by_genre=FALSE, by_distr=FALSE, metric="NONSENSE")

