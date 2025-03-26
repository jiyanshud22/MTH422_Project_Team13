# Set seed for reproducibility
set.seed(123)

# Step 1: Simulate the dataset
n <- 30
p <- 2

# X1 is a vector of ones (intercept)
X1 <- rep(1, n)

# Generate X2 for complete separation at X2 = -0.3
X2 <- c(runif(15, -1, -0.3), runif(15, -0.3, 1))
y <- ifelse(X2 < -0.3, 0, 1)

# Introduce overlap at X2 = 0 to ensure it's not a solitary separator
X2[14:15] <- runif(2, -0.1, 0.1)
X2[29:30] <- runif(2, -0.1, 0.1)
y[14:15] <- 0
y[29:30] <- 1

# Create the design matrix
X <- matrix(c(X1, X2), nrow = n, ncol = p)

# Scatter plot (Figure 5)
# Display the plot first
plot(X2, y, xlab = "X2", ylab = "y", main = "Scatter Plot of y vs X2", pch = 4)
abline(v = -0.3, col = "black", lwd = 2)  # Solid line at -0.3
abline(v = 0, col = "blue", lty = 2, lwd = 2)  # Dashed line at 0

# Save the plot to a file
dev.copy(png, filename = "figure5.png", width = 480, height = 480)
dev.off()  # Close the device after saving

# Step 2: Gibbs Sampler
# Number of iterations
n_iter <- 50000

# Function to compute log-likelihood
log_likelihood <- function(beta, X, y) {
  eta <- X %*% beta
  p <- 1 / (1 + exp(-eta))
  sum(dbinom(y, 1, p, log = TRUE))
}

# Metropolis-Hastings within Gibbs
gibbs_sampler <- function(X, y, prior, n_iter, proposal_sd = 0.5) {
  p <- ncol(X)
  beta_samples <- matrix(0, nrow = n_iter, ncol = p)
  beta <- rep(0, p)  # Initial value
  
  for (iter in 1:n_iter) {
    for (j in 1:p) {
      # Current beta
      beta_current <- beta
      
      # Propose new beta_j
      beta_proposed <- beta
      beta_proposed[j] <- beta[j] + rnorm(1, 0, proposal_sd)
      
      # Log-likelihood
      ll_current <- log_likelihood(beta_current, X, y)
      ll_proposed <- log_likelihood(beta_proposed, X, y)
      
      # Log-prior
      if (prior == "cauchy") {
        lp_current <- sum(dcauchy(beta_current, 0, 1, log = TRUE))
        lp_proposed <- sum(dcauchy(beta_proposed, 0, 1, log = TRUE))
      } else if (prior == "t7") {
        lp_current <- sum(dt(beta_current, df = 7, log = TRUE))
        lp_proposed <- sum(dt(beta_proposed, df = 7, log = TRUE))
      } else {  # normal
        lp_current <- sum(dnorm(beta_current, 0, 1, log = TRUE))
        lp_proposed <- sum(dnorm(beta_proposed, 0, 1, log = TRUE))
      }
      
      # Log-posterior ratio
      log_ratio <- (ll_proposed + lp_proposed) - (ll_current + lp_current)
      
      # Accept or reject
      if (log(runif(1)) < log_ratio) {
        beta[j] <- beta_proposed[j]
      }
    }
    beta_samples[iter, ] <- beta
  }
  return(beta_samples)
}

# Run Gibbs sampler for each prior
beta_cauchy <- gibbs_sampler(X, y, "cauchy", n_iter)
beta_t7 <- gibbs_sampler(X, y, "t7", n_iter)
beta_normal <- gibbs_sampler(X, y, "normal", n_iter)

# Compute running means
running_mean <- function(x) {
  cumsum(x) / (1:length(x))
}

# Plot running means (Figure 4)
# Display the plot first
par(mfrow = c(2, 3))

# Beta 1
plot(running_mean(beta_cauchy[, 1]), type = "l", xlab = "Iterations", ylab = "Beta 1", main = "Cauchy prior")
plot(running_mean(beta_t7[, 1]), type = "l", xlab = "Iterations", ylab = "Beta 1", main = "t7 prior")
plot(running_mean(beta_normal[, 1]), type = "l", xlab = "Iterations", ylab = "Beta 1", main = "normal prior")

# Beta 2
plot(running_mean(beta_cauchy[, 2]), type = "l", xlab = "Iterations", ylab = "Beta 2", main = "Cauchy prior")
plot(running_mean(beta_t7[, 2]), type = "l", xlab = "Iterations", ylab = "Beta 2", main = "t7 prior")
plot(running_mean(beta_normal[, 2]), type = "l", xlab = "Iterations", ylab = "Beta 2", main = "normal prior")

# Save the plot to a file
dev.copy(png, filename = "figure4.png", width = 960, height = 640)
dev.off()  # Close the device after saving

# Print a message to confirm
cat("Plots have been displayed and saved as figure4.png and figure5.png\n")