import numpy as np
import matplotlib.pyplot as plt

print("Hello, World!")
print("信号与系统学习")

# 生成 x 数据
x = np.linspace(0, 2 * np.pi, 100)  # 100 个点

# 显示一个基本的连续信号
# 计算 y = sin(x)
y = np.sin(x)

# 显示一个基本的离散信号
x_discrete = np.arange(0, 2 * np.pi, 0.5)  # 离散点
y_discrete = np.sin(x_discrete)

# 将连续/离散信号画到同一张图的两个子图中
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=False)

# 子图 1：连续信号
axes[0].plot(x, y, label='sin(x)')
axes[0].set_title('y = sin(x) (Continuous)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend()
axes[0].grid(True)

# 子图 2：离散信号
axes[1].stem(x_discrete, y_discrete, label='sin(x) (Discrete)')
axes[1].set_title('y = sin(x) (Discrete)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()
plt.show()