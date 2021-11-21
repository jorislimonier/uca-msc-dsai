# confidence intervals

alpha <- 0.05 # 1-alpha/2 = 0.975
z <- qnorm(1-alpha/2)
N <- 100
p <- .4

rep_exp <- function(){
    obs <- as.numeric(runif(100) < p)

    p_hat <- mean(obs)
    se_hat <- sqrt(p_hat*(1-p_hat)/N)
    half_width <- z*se_hat
    lb <- p_hat - half_width
    ub <- p_hat + half_width
    C_N <- c(lb, ub)

    return(C_N[1] < p && C_N[2] > p_hat)

}

n_runs <- 10000
res <- vector(length=n_runs)
for (run in 1:n_runs) res[run] <- rep_exp()

mean(res)
cat("Confidence interval:", C_N)
