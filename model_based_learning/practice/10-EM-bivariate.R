library(mvtnorm) #library(MASS)

set.seed(10)

###DATA

n <- 500
mu <- c(5, -1)
Sigma <- matrix(c(1, 0.5,0.5,1), nrow=2)
X <- rmvnorm(n, mean=mu, sigma=Sigma) #multivariate Gaussian variable

###INTRODUCE MISSING DATA (MCAR)

prop.miss <- 0.3 #we will introduce 30% of missing values.
nb.miss <- floor(n*prop.miss) #we will introduce 30% of missing values on the second variable.

missing_idx.mcar <- sample(n, nb.miss, replace = FALSE) #nb.miss values at random which will be missing
XNA.mcar <- X
XNA.mcar[1:nb.miss, 2] <- NA


###INITIALIZATION FOR EM

init_EM <- function(XNA){ #it returns the initial value for theta=(mu,sigma)
  mu1 <- mean(XNA[,1])
  mu2 <- mean(XNA[,2],na.rm=TRUE)
  sigma <- cov(XNA.mcar,use="complete.obs")
  mu <- c(mu1,mu2)
  return(list(mu_init=mu,sigma_init=sigma))
}

e_step <- function(XNA,mu_r,sigma_r){ #e-step for iteration r
  

  
  return(list(s1=,s11=,s2=,s22=,s12=))
}

m_step <- function(){
  
  
}
