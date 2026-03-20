import numpy as np
import matplotlib.pyplot as plt

print("Hello, World!")
print("信号与系统学习")

# 生成 x 数据
x = np.linspace(0, 2 * np.pi, 100)  # 100 个点

# 计算 y = sin(x)
y = np.sin(x)

# 绘制 sin(x) 图像
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='sin(x)')
plt.title('y = sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()