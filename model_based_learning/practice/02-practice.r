library(MASS)
?lda
X = iris[,1:4]
Y = iris$Species
f.lda = lda(X,Y)
f.qda = qda(X,Y)
f.qda
f.lda$scaling
y_star = predict(f.lda, X)
y_star

## V-fold cross validation
V = 15
n_samples = nrow(X)
folds = rep(1:V, n_samples/V)
err.lda = rep(NA, V)
err.qda = rep(NA, V)

for (v in 1:V){
    X.learn = X[folds!=v,]
    Y.learn = Y[folds!=v]
    X.val = X[folds==v,]
    Y.val = Y[folds==v]

    # LDA
    # classifier
    f.lda = lda(X.learn, Y.learn)

    # prediction
    yhat.lda = predict(f.lda, X.val)$class
    
    # error
    err.lda[v] = sum(yhat.lda != Y.val) / length(Y.val)
    
    # LDA
    # classifier
    f.qda = qda(X.learn, Y.learn)

    # prediction
    yhat.qda = predict(f.qda, X.val)$class
    
    # error
    err.qda[v] = sum(yhat.qda != Y.val) / length(Y.val)
}

# Results
cat("LDA:\t", mean(err.lda), sd(err.lda))
cat("QDA:\t", mean(err.qda), sd(err.qda))

# KNN
library(class)
x_star = c(5.5, 3, 4, 1.5)
y_hat = knn(X, x_star, Y, k=3)


V = 15
K = 25 # nb of neighbors

n_samples = nrow(X)
folds = rep(1:V, n_samples/V)

err.knn = matrix(NA, V, K)

for (v in 1:V){
    X.learn = X[folds!=v,]
    Y.learn = Y[folds!=v]
    X.val = X[folds==v,]
    Y.val = Y[folds==v]
    for (k in 1:K){
        y_hat = knn(X.learn, X.val, Y.learn, k=k)
        err.knn[v,k] = sum(y_hat != Y.val) / length(Y.val)    
    }
}
plot(1:K, colMeans(err.knn), type="b")
