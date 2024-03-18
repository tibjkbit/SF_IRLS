# 安装必要的包
if (!requireNamespace("MASS", quietly = TRUE)) install.packages("MASS")
library(MASS)
library(splines)
library(ggplot2)

# 加载数据
data <- read.csv("data.txt", header = FALSE, sep = " ")
names(data) <- c("x", "y")

x <- data$x
y <- data$y

# 定义B样条基函数
p <- 10  # 基函数的数量
Phi <- bs(x, df = p)
X <- cbind(1, Phi)  # 添加截距项

# Step 1: 初始值设定
beta_hat <- lm(y ~ X - 1)$coefficients  # 无截距的线性模型
sigma_hat_squared <- sum((y - X %*% beta_hat)^2) / length(y)

# 使用pnorm和dnorm获取F*和f*的函数
get_lambda <- function(y, X, beta, sigma_squared) {
    epsilon <- (y - X %*% beta) / sqrt(sigma_squared)
    lambda <- sum(dnorm(epsilon) / (1 - pnorm(epsilon))) / sum((y - X %*% beta)^2)
    return(lambda)
}

lambda_hat <- get_lambda(y, X, beta_hat, sigma_hat_squared)

# 设置步长，这可能需要根据实际情况调整
alpha <- 0.0002

# Step 2: 利用梯度更新beta_hat中的截距项
convergence <- FALSE
max_iter <- 50000  # 设置最大迭代次数
iter <- 0

while(!convergence && iter < max_iter) {
    iter <- iter + 1
    
    # 计算梯度
    epsilon <- (y - X %*% beta_hat) / sqrt(sigma_hat_squared)
    gradient <- t(X) %*% (y - X %*% beta_hat) / sigma_hat_squared + 
        lambda_hat / sqrt(sigma_hat_squared) * t(X) %*% (dnorm(epsilon) / (1 - pnorm(epsilon)))
    
    # 只更新beta_0 (截距项)
    beta_hat[1] <- beta_hat[1] + alpha * gradient[1]
    
    # 重新计算sigma_hat_squared和lambda_hat
    sigma_hat_squared <- sum((y - X %*% beta_hat)^2) / length(y)
    lambda_hat <- get_lambda(y, X, beta_hat, sigma_hat_squared)
    
    # 检查收敛条件
    if(abs(gradient[1]) < 1e-6) {
        convergence <- TRUE
        cat("Convergence reached at iteration", iter, "\n")
    }
}

# Step 3: 检查收敛或最大迭代次数
if (convergence) {
    cat("Algorithm converged in", iter, "iterations.\n")
} else {
    cat("Algorithm did not converge in the maximum number of iterations.\n")
}

# 输出结果
results <- list(beta_hat = beta_hat, sigma_hat_squared = sigma_hat_squared, lambda_hat = lambda_hat)

# 计算拟合值，注意我们在这里使用完整的X（包括截距项和基函数）
fitted_values <- X %*% beta_hat

# 准备绘图数据
plot_data <- data.frame(x = x, y = y, fitted = fitted_values)

# 绘制散点图和拟合曲线
ggplot(plot_data, aes(x = x, y = y)) +
    geom_point() +
    geom_line(aes(y = fitted), color = 'red') +
    theme_minimal() +
    ggtitle("Stochastic Frontier Model: Scatter Plot with Fitted Curve") +
    theme(plot.title = element_text(hjust = 0.5))  # 居中标题

