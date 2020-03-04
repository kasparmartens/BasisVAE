library(tidyverse)

generate_toy_data <- function(z, p, scaling, noise_sd = 0.05){
  n_cols <- 4L*p
  N <- length(z)
  Y <- matrix(0, N, n_cols)
  
  transient <- function(x) (exp(-0.5*(x**2)))
  softplus <- function(x) 0.4*log(exp(x) + 1)
  
  for(j in 1:p){
    Y[, j+0*p] <- scaling[j] * transient(z+1.5)
  }
  for(j in 1:p){
    Y[, j+1*p] <- scaling[j] * transient(z)
  }
  for(j in 1:p){
    Y[, j+2*p] <- scaling[j] * transient(z-1.5)
  }
  for(j in 1:p){
    Y[, j+3*p] <- scaling[j] * softplus(z)
  }
  
  Y <- Y + noise_sd * rnorm(N*n_cols)
  
  colnames(Y) <- sprintf("feature%d", 1:ncol(Y))

  data.frame(Y)
}

# sample size
N <- 500L
# number of repeated feature shapes
p <- 10L

set.seed(0)
z <- runif(N, -3, 3)
lambda <- seq(0.4, 1.6, length=p)
Y <- generate_toy_data(z, p, lambda, noise_sd=0.2)

write_csv(round(Y, 5), "toy_data/toy.csv")



### For visualisation purposes generate z on a grid

z_grid <- seq(-3, 3, length=200)

Y_grid <- generate_toy_data(z_grid, p, lambda, noise_sd=0.0)

Y_grid %>%
  mutate(z = z_grid) %>%
  gather(variable, value, -z) %>%
  mutate(variable = forcats::fct_inorder(factor(variable))) %>%
  mutate(variable_group = (as.numeric(variable)-1L) %/% p) %>%
  ggplot(aes(z, value, col=factor(variable_group), group=variable)) +
  geom_path() +
  facet_wrap(~ variable_group, nrow=1) +
  theme_classic() +
  theme(legend.position = "none")

