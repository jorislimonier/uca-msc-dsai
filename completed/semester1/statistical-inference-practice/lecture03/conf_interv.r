library(bootstrap)
data("law")

lsat <- law$LSAT
gpa <- law$GPA

rho_ <- cor(lsat, gpa)
plot(gpa, lsat)

B <- 1000
store_rho_b <- vector(length=B)
for (b in 1:B) {
    pos <- sample(1:15, replace=TRUE)
    obs_b <- law[pos,]
    lsat_b <-obs_b$LSAT
    gpa_b <-obs_b$GPA
    rho_b <- cor(lsat_b, gpa_b)
    store_rho_b[b] <- rho_b
}
se_b <- sd(store_rho_b)

# Build a 95% bootstrap CI
CI_N <- c(rho_ + qnorm(.025)*se_b, rho_ + qnorm(.975)*se_b)
CI_pi <- c(2*rho_ - quantile(store_rho_b, .975), 2*rho_ - quantile(store_rho_b, .025))
CI_pe <- c(quantile(store_rho_b, .025), quantile(store_rho_b, .975))

print(mean(store_rho_b))


length(lsat)
