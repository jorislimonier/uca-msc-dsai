LSM <- function(A,dim=2,...){
   # A is the adjacency matrix and is denoted by X in the maths,
   # dim is the dimensionality of the latent space (2 by default)
   n = nrow(A)
   
   # definition of the log-likelihood to optimize
   negloglik <- function(theta,A,dim){
      ll = 0
      alpha = theta[1]
      Z = matrix(theta[-1],ncol=dim)
      D = as.matrix(dist(Z))
      for (i in 1:(n-1)){
         for (j in (i+1):n){
            ll = ll + A[i,j]*(alpha - D[i,j]) - log(1 + exp(alpha - D[i,j]))
         }
      }
      negll = - 2 * ll
   }
   
   # Initial values for theta
   alpha_0 = 0.5
   Z_0 = rnorm(n*dim,0,1)
   theta = c(alpha_0,Z_0)
   
   # Numerical optimization
   theta_opt = optim(theta,negloglik,A=A,dim=dim,...) 
   
   # Plot and return the results
   alpha = theta_opt$par[1]
   Z = matrix(theta_opt$par[-1],ncol=dim)
   return(list(alpha=alpha,Z=Z,Zinit=matrix(Z_0,ncol=dim)))
}

X = rbind(c(0,1,1,1,1),
          c(1,0,0,0,0),
          c(1,0,0,0,0),
          c(1,0,0,0,0),
          c(1,0,0,0,0))

out = LSM(X,dim=2,method="SANN")
par(mfrow=c(1,2))
gplot(X,coord = out$Zinit,edge.col = "lightgray")
gplot(X,coord = out$Z,edge.col = "lightgray")
