import numpy as np

# Sigmoid激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # 注意：这里输入是sigmoid的输出

# 初始化参数
w1 = 0.2
b1 = 0.1
w2 = 0.5
b2 = 0.3

# 输入和目标值
x = 1.0
y = 0.5

# 学习率
learning_rate = 0.1

print("初始参数: w1 = {:.4f}, b1 = {:.4f}, w2 = {:.4f}, b2 = {:.4f}".format(w1, b1, w2, b2))
print("输入: x = {:.1f}, 目标值: y = {:.1f}\n".format(x, y))

# 正向传播
z1 = w1 * x + b1
a1 = sigmoid(z1)
z2 = w2 * a1 + b2
a2 = sigmoid(z2)

print("正向传播:")
print("隐藏层加权输入 z1 = {:.4f}".format(z1))
print("隐藏层激活输出 a1 = {:.4f}".format(a1))
print("输出层加权输入 z2 = {:.4f}".format(z2))
print("输出层激活输出 a2 = {:.4f}".format(a2))
print("损失 L = {:.6f}\n".format(0.5 * (y - a2) ** 2))

# 反向传播
# 输出层误差
delta2 = (a2 - y) * sigmoid_derivative(a2)
# 输出层参数梯度
dw2 = delta2 * a1
db2 = delta2

# 隐藏层误差
delta1 = delta2 * w2 * sigmoid_derivative(a1)
# 隐藏层参数梯度
dw1 = delta1 * x
db1 = delta1

print("反向传播:")
print("输出层误差 delta2 = {:.6f}".format(delta2))
print("w2 梯度 = {:.6f}, b2 梯度 = {:.6f}".format(dw2, db2))
print("隐藏层误差 delta1 = {:.6f}".format(delta1))
print("w1 梯度 = {:.6f}, b1 梯度 = {:.6f}\n".format(dw1, db1))

# 参数更新
w1_new = w1 - learning_rate * dw1
b1_new = b1 - learning_rate * db1
w2_new = w2 - learning_rate * dw2
b2_new = b2 - learning_rate * db2

print("更新后参数:")
print("w1 = {:.4f}, b1 = {:.4f}, w2 = {:.4f}, b2 = {:.4f}".format(w1_new, b1_new, w2_new, b2_new))

# 验证更新后的预测结果
z1_new = w1_new * x + b1_new
a1_new = sigmoid(z1_new)
z2_new = w2_new * a1_new + b2_new
a2_new = sigmoid(z2_new)

print("\n更新后预测:")
print("输出 a2 = {:.4f}".format(a2_new))
print("更新后损失 L = {:.6f}".format(0.5 * (y - a2_new) **2))
