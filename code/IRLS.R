library(splines)
library(ggplot2)

# 加载数据
data <- read.csv("data.txt", header = FALSE, sep = " ")
names(data) <- c("x", "y")

x <- data$x
y <- data$y

# 定义B样条基函数
p <- 10 # 基函数的数量
Phi <- bs(x, df = p)
Phi <- cbind(1, Phi)  # 添加截距项

# 无约束最小二乘法估计作为初始值
beta_hat <- solve(t(Phi) %*% Phi) %*% t(Phi) %*% y
beta_hat <- beta_hat - max(beta_hat)


# 检查约束是否满足
constraints_satisfied <- all(y - Phi %*% beta_hat >= 0)
cat("First Constraints satisfied:", constraints_satisfied, "\n")

# 设置收敛阈值和迭代最大次数
threshold <- 1e-6
max_iter <- 50

# 计算初始r值
u <- y - Phi %*% beta_hat
v <- 1 / (y - Phi %*% beta_hat)
r <- sum(u * (Phi %*% t(Phi)) %*% v) / sum(v * (Phi %*% t(Phi)) %*% v)
r <- ifelse(r > 0, r, 1)

cat("The r_0:", r)


for (iter in 1:max_iter) {
    Phi_beta <- Phi %*% beta_hat
    inv_Phi_beta <- 1 / Phi_beta  # 逐元素求倒数
    
    residuals <- y - Phi_beta
    
    
    
    # 计算S(beta)
    S_beta <- -t(Phi) %*% residuals + r * colSums(Phi / matrix(Phi_beta, nrow = nrow(Phi), ncol = ncol(Phi), byrow = TRUE))
    
    # 计算W的对角线元素
    W_diag <- -1 - r / (residuals^2)
    W <- diag(as.vector(W_diag))  # 创建对角矩阵W
    
    # 计算Z(beta)
    Z <- Phi_beta + Phi %*% solve(t(Phi) %*% W %*% Phi) %*% S_beta
    
    # 更新beta
    beta_new <- solve(t(Phi) %*% W %*% Phi) %*% t(Phi) %*% W %*% Z
    
    # 检查收敛性
    predictions <- Phi %*% beta_new
    residuals <- y - predictions
    min_residuals <- min(residuals)
    
    if(min_residuals < 0) {
        cat("Convergence achieved after", iter, "iterations.\n")
        break
    }
    
    
    
    # 更新r值
    r <-  r / 1.5 # 可选的衰减
    
    # 更新beta_hat
    beta_hat <- beta_new
    
    # 报告当前迭代信息
    cat("Iteration:", iter, "- r value:", r, "\n")
    
    # 检查约束是否满足
    constraints_satisfied <- all(Phi %*% beta_hat >= 0)
    cat("Constraints satisfied:", constraints_satisfied, "\n")
}
# 输出最终估计
print(beta_hat)

# 绘制结果
fitted_values <- Phi %*% beta_hat

plot_data <- data.frame(x = x, y = y, fitted = fitted_values)

# 绘制散点图和拟合曲线
ggplot(plot_data, aes(x = x, y = y)) +
    geom_point() +
    geom_line(aes(y = fitted), color = 'red') +
    theme_minimal() +
    ggtitle("Scatter Plot with Fitted Curve")
