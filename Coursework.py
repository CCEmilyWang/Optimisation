## Part B 
import numpy as np
import time

# 定义问题参数
lambda_ = 5e-4
gamma = 1e3
omega = 75
max_iter = 500
tol_u = 1e-4
tol_x = 1e-5

# 定义h_gamma函数
def h_gamma(t, gamma):
    if abs(t) <= 1 / gamma:
        return 0.5 * gamma * t**2
    else:
        return abs(t) - 1 / (2 * gamma)

# 定义近似l1范数的梯度
def grad_l1_approx(u, gamma, omega):
    grad = np.zeros_like(u)
    for i in range(len(u)):
        if abs(u[i]) <= 1 / gamma:
            grad[i] = omega[i] * gamma * u[i]
        else:
            grad[i] = omega[i] * np.sign(u[i])
    return grad

# 定义梯度下降法
def gradient_descent(A, L, f, u, z, gamma, lambda_, omega):
    for k in range(max_iter):
        grad = A.T @ (A @ u - f) + lambda_ * L.T @ (L @ u - z) + grad_l1_approx(u, gamma, omega)
        
        # 回溯线搜索
        alpha = 1
        while True:
            u_new = u - alpha * grad
            if objective(A, L, u_new, f, z, gamma, lambda_, omega) <= objective(A, L, u, f, z, gamma, lambda_, omega) - alpha * np.dot(grad, grad):
                break
            alpha *= 0.5
        
        # 更新u
        u = u_new
        
        # 检查停止准则
        if np.linalg.norm(u_new - u) / np.linalg.norm(u_new) < tol_u:
            break

    return u

# 计算目标函数（示意）
def objective(A, L, u, f, z, gamma, lambda_, omega):
    return 0.5 * np.linalg.norm(A @ u - f)**2 + 0.5 * lambda_ * np.linalg.norm(L @ u - z)**2 + np.sum([omega[i] * h_gamma(u[i], gamma) for i in range(len(u))])

# 执行梯度下降
start_time = time.time()
u_optimized = gradient_descent(A, L, f, u_init, z_init, gamma, lambda_, omega)
end_time = time.time()

# 输出结果
print(f"Optimized u: {u_optimized}")
print(f"Execution time: {end_time - start_time} seconds")
