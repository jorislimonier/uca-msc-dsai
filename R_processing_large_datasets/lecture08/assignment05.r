# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
library(dplyr)
library(ggplot2)
library(gridExtra)
library(rpart)
library(rpart.plot)

# %% [markdown]
# # Supervised model

# %%
initial_adult <- read.csv("./datasets/adult.csv")


# %%
adult <- select(initial_adult, -c(x, educational.num))
head(adult)


# %%
summary(adult)


# %%
ggplot(adult) + aes(x=as.numeric(age), group=income, fill=income) + 
  geom_histogram(binwidth=1, color="black")
ggplot(adult) + aes(x=as.numeric(age), group=gender, fill=gender) + 
  geom_histogram(binwidth=1, color='black')
ggplot(adult) + aes(x=as.numeric(age), group=race, fill=race) + 
  geom_histogram(binwidth=1, color='black')
ggplot(adult) + aes(x=as.numeric(hours.per.week), group=income, fill=income) + 
  geom_histogram(binwidth=3, color='black') +
  scale_y_log10()


# %%
sum(is.na(adult))

# %% [markdown]
# There is no NA, but we want to investigate whether there are missing values categorized as some other way.

# %%
for (col in colnames(adult)){
    print(c(unique(adult[col])))
}

# %% [markdown]
# There are some `?` values in the `workclass` column and `Other` in the `race` column.

# %%
sum(adult$workclass == "?") # number of `?`
sum(adult$race == "Other") # number of `Other`

# %% [markdown]
# Drop "?" and "other" observations

# %%
adult <- adult[!adult$workclass == "?",]
adult <- adult[!adult$race == "Other",]

# %% [markdown]
# Now label encode categorical values before feeding

# %%
for (col in c("workclass", "education", "marital.status", "race", "gender", "income")){
    adult[[col]] <- as.integer(factor(adult[[col]], labels=1:length(unique(adult[[col]]))))-1
}


# %%
create_train_test <- function(data, size=0.8, train=TRUE, seed=TRUE){
    if (seed) {
        set.seed(42)
    }
    smp_size <- floor(size * nrow(data))
    train_ind <- sample(seq_len(nrow(data)), size = smp_size)
    
    if (train) {
        return (data[train_ind, ])
    } else {
        return (data[-train_ind, ])
    }
}
data_train <- create_train_test(adult, size=0.8, train=TRUE)
data_test <- create_train_test(adult, size=0.8, train=FALSE)
# X_train <- select(data_train, -income)
# y_train <- select(data_train, income)
# X_test <- select(data_test, -income)
# y_test <- select(data_test, income)


# %%
dim(data_train)
dim(data_test)


# %%
prop.table(table(data_train$income))
prop.table(table(data_test$income))

# %% [markdown]
# ### Plot decision tree

# %%
fit <- rpart(income~., data_train, method="class")
rpart.plot(fit, extra=106)

# %% [markdown]
# ### Display confusion

# %%
pred <- predict(fit, data_test, type="class")
conf_mat <- table(pred, data_test$income)
print(conf_mat)

# %% [markdown]
# ### Deduce model accuracy

# %%
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
print(paste("accuracy:", accuracy))

# %% [markdown]
# ### Display model parameters

# %%
rpart.control()


# %%
control <- rpart.control()
fit <- rpart(income~., data_train, method="class", control=control)
pred <- predict(fit, data_test, type="class")
conf_mat <- table(pred, data_test$income)
print(conf_mat)
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
print(paste("accuracy:", accuracy))

# %% [markdown]
# # Unsupervised model

# %%
library(cluster)
library(factoextra)
library(magrittr)


# %%
initial_cars <- mtcars
cars <- data.frame(initial_cars)
head(initial_cars)


# %%
sum(is.na(cars_data)) # no NA


# %%
for (col in colnames(cars)){
    print(c(unique(cars[col])))
}


# %%
cars <- data.frame(scale(cars))
head(cars)
cars <- select(cars, -c(mpg))


# %%
km.res <- kmeans(cars, 3, nstart=25)


# %%
fviz_cluster(km.res, data=cars, ellipse.type="convex")


# %%
pam.res <- pam(cars, 3)
fviz_cluster(pam.res) #almost similar to kmeans results

# %% [markdown]
# We see that for the most part, K-means and PAM classify in the same way. Only some samples between the center and the upper-right classes change class when modifying the method.

# %%



# %%



