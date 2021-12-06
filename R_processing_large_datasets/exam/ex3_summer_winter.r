library(dplyr)
library(ggplot2)
library(gridExtra)

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
hist_summer_played <- swo %>%
    ggplot(aes(summer_played)) +
    geom_histogram(bins = 20)
hist_summer_played

# Question 4b
hist_winter_played <- swo %>%
    ggplot(aes(winter_played)) +
    geom_histogram(bins = 20)
hist_winter_played

# Question 4c
grid.arrange(hist_summer_played, hist_winter_played)

# Question 4d

hist_summer_total <- swo %>%
    ggplot(aes(summer_total)) +
    geom_histogram(bins = 20)
hist_summer_total

hist_winter_total <- swo %>%
    ggplot(aes(winter_total)) +
    geom_histogram(bins = 20)
hist_winter_total
grid.arrange(hist_summer_total, hist_winter_total)

grid.arrange(
    hist_summer_played, hist_winter_played,
    hist_summer_total, hist_winter_total
)

# Question 4e
print(
    paste(
        "The correlation between total number of",
        "medals won in summer and in winter is:",
        cor(swo$summer_total, swo$winter_total)
    )
)
swo %>%
    ggplot(aes(summer_total, winter_total)) +
    geom_point() +
    stat_smooth(method = "lm")



# Question 4f
print(
    paste(
        "The correlation between total number of",
        "games played in summer and in winter is:",
        cor(swo$summer_played, swo$winter_played)
    )
)
swo %>%
    ggplot(aes(summer_played, winter_played)) +
    geom_point() +
    stat_smooth(method = "lm")

# Question 4g
hist_summer_gold <- swo %>% ggplot(aes(summer_gold)) +
    geom_histogram(bins = 20)
hist_summer_silver <- swo %>% ggplot(aes(summer_silver)) +
    geom_histogram(bins = 20)
hist_summer_bronze <- swo %>% ggplot(aes(summer_bronze)) +
    geom_histogram(bins = 20)
hist_winter_gold <- swo %>% ggplot(aes(winter_gold)) +
    geom_histogram(bins = 20)
hist_winter_silver <- swo %>% ggplot(aes(winter_silver)) +
    geom_histogram(bins = 20)
hist_winter_bronze <- swo %>% ggplot(aes(winter_bronze)) +
    geom_histogram(bins = 20)

grid.arrange(
    hist_summer_gold,
    hist_summer_silver,
    hist_summer_bronze,
    hist_winter_gold,
    hist_winter_silver,
    hist_winter_bronze
)


# Question 4h
hist_summer_gold <- swo %>% ggplot(aes(summer_gold)) +
    geom_histogram(bins = 10)
hist_summer_silver <- swo %>% ggplot(aes(summer_silver)) +
    geom_histogram(bins = 10)
hist_summer_bronze <- swo %>% ggplot(aes(summer_bronze)) +
    geom_histogram(bins = 10)
hist_winter_gold <- swo %>% ggplot(aes(winter_gold)) +
    geom_histogram(bins = 10)
hist_winter_silver <- swo %>% ggplot(aes(winter_silver)) +
    geom_histogram(bins = 10)
hist_winter_bronze <- swo %>% ggplot(aes(winter_bronze)) +
    geom_histogram(bins = 10)

grid.arrange(
    hist_summer_gold,
    hist_summer_silver,
    hist_summer_bronze,
    hist_winter_gold,
    hist_winter_silver,
    hist_winter_bronze
)

# Question 4i
install.packages("ggcorrplot")
library(ggcorrplot)

install.packages("GGally")
library(GGally)

install.packages("wordcloud")
library(wordcloud)

numcol <- swo %>%
    colnames() %>%
    tail(-2)

swo %>%
    select(all_of(numcol)) %>%
    cor() %>%
    ggcorrplot()

swo %>%
    ggparcoord(columns = 3:17, groupColumn = 3) +
    scale_x_discrete(guide = guide_axis(angle = 45))

swo %>% ggplot(aes(summer_played)) +
    geom_boxplot()

wordcloud(
    swo$NOC,
    swo$summer_played,
    max.words = 50,
    rot.per = .35,
    min.freq = 10,
    random.order = FALSE,
    colors = brewer.pal(8, "Dark2")
) # I tried, but the wordcloud doesn't seem to work well