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

for (v in 1:V){
    X.learn = X[folds!=v,]
    Y.learn = Y[folds!=v]
    X.val = X[folds==v,]
    Y.val = Y[folds==v]
}
X.learn
folds
