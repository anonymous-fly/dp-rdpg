pkgs <- c("tidyverse","network","Rdimtools","mvtnorm",
          "rgl","TDA","Riemann","scatterplot3d", "RSpectra")
# sapply(pkgs, install.packages, character.only=T)
sapply(pkgs,require,character.only=T)



# Functions ####


# 1(A): Some plot functions. Really bad hack. Please ignore. ----

plt.net <- function(Net,...){
  plot.network(network(Net,directed = F),...)
}


plt3d <- function(x, add=F, angle=60, alpha=0.5, col="black",L=3){
  if(length(L)<2){L <- c(L,L)}
  if(length(L)==2){
    lims <- matrix(rep(c(-L[1],L[2]),3),nrow = 2)
  } else
  {
    lims <- cbind(L[1:2],L[3:4],L[5:6])
  }
  scatterplot3d(x, angle=angle, color=alpha(col,alpha), pch = 20,
                xlab="x", ylab="y", zlab="z",
                xlim = lims[,1], ylim=lims[,2],zlim=lims[,3])
}


# 1(B): Functions for making the adjacency matrix and spectral embedding ----

symmetrize <- function(m) {
  m[upper.tri(m)] <- t(m)[upper.tri(m)]
  m
}

Adjacency <- function(f,Z){
  if(is.matrix(Z)){
    n <- nrow(Z)
    p <- f(Z %*% t(Z))
  }
  else{
    n <- length(Z)
    p <- outer(Z, Z, FUN = f)
  }
  X <- matrix(0,n,n)
  X[lower.tri(X)] <- rbernoulli(sum(lower.tri(p)), p[lower.tri(p)])
  return(symmetrize(X))
}

Laplacian <- function(A){
  # D <- diag(rowSums(A)-1)
  D <- (1/(rowSums(A)-1)) %>% sqrt() %>% diag()
  return( diag(rep(1,nrow(A))) - D%*%A%*%D)
}

Spectral <- function(L,d=3){
  E = eigs_sym(L, d)
  X = E$vectors %*% diag(sqrt(abs(E$values)))
  return(X)
}

edgeFlip <- function(Adj,e=NA,p=0.01){
  if(!is.na(e)){
    p <- 1/(1+exp(e))
  }
  ind <- lower.tri(Adj)
  flips <- rbernoulli(sum(ind),p=p)
  Adj[ind] <- ifelse(flips,1-Adj[ind],Adj[ind])
  Adj <- symmetrize(Adj)
  return(Adj)
}

geodesic_circle <- function(x,y,r=1){
  return(r*acos((t(x)%*%y)/(r^2)))
}

geodesic_sphere <- function(x,y,r=1){
  return(r*acos((t(x)%*%y)/(1.01*r^2)))
}

privacy <- function(p=NA,e=NA){
  flag1 <- is.na(e)
  flag2 <- is.na(p)
  if(flag1 & flag2){
    warning("Ooowee, looks like you've forgotten something!")
  } 
  if(flag1){
    return(log((1-p)/p))
  }
  if(flag2){
    return(1/(1+exp(e)))
  }
}


# 2. Examples ####

# Example 0
if(TRUE){
  set.seed(2020)
  n <- 500
  p <- 0.1
  f <- function(x,y){(0.5)+(p)*(x==y) - p*(x!=y)}
  Z <- sample(c(-1,0,1),n,replace = T)
  A <- Adjacency(f,Z)
  X <- A %>%  Spectral() %>% scale()
  # rgl::plot3d(X[,1:3])
}

# Example 1
if(FALSE){
  set.seed(2020)
  n <- 1000
  f <- function(x,y){1-exp(-2*x*y)}
  Z <- rgamma(n,shape = 1, rate = 1)
  A <- Adjacency(f,Z) 
  X <- A %>% Spectral() %>% scale()
  rgl::plot3d(X[,1:3] %>% scale())
}




# # Example 2
if(FALSE){
  set.seed(2020)
  n <- 500
  f <- function(x,y,s=0.05){exp(-((x-y)^2)/(2*(s^2)))}
  Z <- runif(n,-1*pi,1*pi)
  A <- Adjacency(f,Z)
  X <- A %>% Spectral()
  rgl::plot3d(X[,1:3])
}




# Example 3
if(FALSE){
  set.seed(2020)
  n <- 500
  Z <- TDA::circleUnif(n)
  f <- function(x,y){dnorm(geodesic_circle(x,y),sd=0.5)}
  A <- Adjacency(f,Z)
  X <- A %>% Spectral() %>% scale()
  rgl::plot3d(X[,1:3] %>% scale())
  
}



# Example 4
if(FALSE){
  
  set.seed(2020)
  n <- 500
  Z <- tdaunif::sample_lemniscate_gerono(n)
  f <- function(x,y){dnorm(sum((x-y)^2),sd = 0.8)}
  A <- Adjacency(f,Z)
  # plt.net(A,edge.lwd=0.01,edge.col=alpha("black",0.01))
  X <- A %>% Spectral()
  rgl::plot3d(X[,1:3] %>% scale())
}



# Example 5
if(FALSE){
  set.seed(2020)
  n <- 400
  Z <- TDA::circleUnif(n)
  Z <- rbind(Z,
             rmvnorm(n/4,c(2,0),0.01*diag(2)),
             # rmvnorm(n/4,c(0,2),0.01*diag(2)),
             # rmvnorm(n/4,c(0,-2),0.01*diag(2)),
             rmvnorm(n/4,c(-2,0),0.01*diag(2)))
  
  plot(Z,asp=1)
  
  f <- function(x,y){
    a <- sum((x)^2) > 1.1
    b <- sum((y)^2) > 1.1
    if(a & b){
      return(0.5+(4.999*sign(x[1]*y[1])*sign(x[2]*y[2])))
    }
    if((a&(!b))|((!a)&b)){
      return(0.1)
    }
    if(!a & !b){
      return(dnorm(geodesic_circle(x,y),sd=1))
    }
  }
  
  A <- Adjacency(f,Z)
  # plt.net(A,edge.lwd=0.01,edge.col=alpha("black",0.05))
  X <- A %>% Spectral()
  rgl::plot3d(X[,1:2] %>% scale(),col=c(rep("red",n),rep("green",n/4),rep("blue",n/4)))
}






## 2. Persistent Homology ####

# Assuming A is already computed
n <- nrow(A)
B <- A %>% edgeFlip(p=privacy(e=2))

X <- A %>% Spectral() %>% scale()
Y <- B %>% Spectral() %>% scale()

dgm1 <- TDA::alphaShapeDiag(X, maxdimension = 2)
dgm2 <- TDA::alphaShapeDiag(Y, maxdimension = 2)




par(mfrow=c(1,3))
# Scatterplot of spectral embedding
plt3d(
  rbind(X[,1:3],Y[,1:3]), 
  angle=30, L=3, 
  col=c(rep('red',n),rep('black',n))
)
title("Spectral Embedding")


# Persistence Diagram for A
plot(dgm1$diagram,main="Non-Private")


# Persistence Diagram for B
dlim <- c(0, tail(sort(dgm1$diagram[,3]),2)[1])
plot(dgm2$diagram, main=paste("Private, ϵ =",privacy(p=p) %>% round(2)), diagLim = dlim)


# Wasserstein distance
TDA::bottleneck(dgm1$diagram, dgm2$diagram, dimension=0)
TDA::wasserstein(dgm1$diagram, dgm2$diagram, dimension=0)
