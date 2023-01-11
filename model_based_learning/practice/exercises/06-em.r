EM_func <- function(X, K, maxit = 50) {
  n <- length(X)
  lik <- rep(0, maxit)
  T <- matrix(NA, nrow = n, ncol = K) # matrix of t_{ik}


  # Initialization of theta (start with E step)
  prop <- rep(1 / K, K) # pi
  sigma_sq <- rep(1, K)
  mu <- rnorm(n = K, mean = mean(X), sd = 1)

  # Main loop of EM
  for (it in 1:maxit) {
    # E step
    for (k in 1:K) {
      T[, k] <- prop[k] * dnorm(X, mean = mu[k], sd = sqrt(sigma_sq[k]))
    }

    T <- T / rowSums(T) %*% matrix(1, nrow = 1, ncol = K) # normalize

    # M step
    for (k in 1:K) {
      prop[k] <- sum(T[, k]) / n
      mu[k] <- sum(T[, k] * X) / sum(T[, k])
      sigma_sq[k] <- sum(T[, k] * (X - mu[k])^2) / sum(T[, k])
    }

    # Visualization
    grp <- max.col(T)
    # plot(X, rep(1:n), col = grp)
    points(mu, rep(1:3), pch = 19, cex = 2, col = 1:3)
    # Sys.sleep(0.5)

    # Likelihood evaluation
    # Use logsumexp, because this is exploding
    for (k in 1:K) {
      lik[it] <- lik[it] + sum(prop[k] * dnorm(X, mu[k], sqrt(sigma[k])))
    }
  }



  return(list(prop = prop, mu = mu, sigma = sigma, T = T, grp = grp, lik=lik))
}
mu <- c(0, 2, -2)
sigma <- c(0.2, 0.3, 0.2)
X <- c(
  rnorm(100, mu[1], sqrt(sigma[1])),
  rnorm(100, mu[2], sqrt(sigma[2])),
  rnorm(100, mu[3], sqrt(sigma[3]))
)
cls <- rep(1:3, rep(100, 3))
out <- EM_func(X = X, K = 3)
plot(X, col = out$grp)
out$lik
