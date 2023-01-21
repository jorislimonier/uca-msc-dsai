log_gaussian_density <- function(x, mean, sigma) {
  distval <- mahalanobis(x, center = mean, cov = sigma)
  logdet <- determinant(sigma, logarithm = TRUE)$modulus
  logretval <- -(ncol(x) * log(2 * pi) + logdet + distval) / 2
  return(logretval)
}

logsumexp <- function(x) {
  y <- max(x)
  return(y + log(sum(exp(x - y))))
}

normalise <- function(x) {
  logratio <- log(x) - logsumexp(log(x))
  return(exp(logratio))
}

em_term <- function(k, X, pi_, mu, sigma) {
  res <- pi_[k] * dnorm(x = X, mean = mu[k], sd = sqrt(sigma[k]))
  print(length(res))
  print(length(gamma[, k]))
  return(res)
}

e_step <- function(X, pi_, mu, sigma) {
  gamma <- matrix(nrow = nrow(X), ncol = ncol(X))

  denom <- 0
  for (k in seq_len(ncol(X))) {
    num <- em_term(k, X, pi_, mu, sigma)
    denom <- denom + num
    gamma[, k] <- num
  }
  gamma <- gamma / denom

  return(gamma)
}
