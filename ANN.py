import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

iris = load_iris()
# print(iris)
# # 读取文件
# df = pd.read_excel(r'E:\数据挖掘\作业9\test_data.xlsx', sheet_name='Sheet2', header=0)

# 调整原始数据格式
data = np.array(iris['data']).reshape(-1, 4)
res = np.array(iris['target']).reshape(-1, 1)
# test = np.array([[1.5]]).reshape(-1, 1)

# 因为lbfgs在小数据集上收敛更快所以solver选用lbfgs
adam_model = MLPClassifier(hidden_layer_sizes=(100,), learning_rate="adaptive", solver='adam', max_iter=5000)

# 训练预测输出结果
adam_model.fit(data, res)
loss_values = adam_model.loss_curve_

lbfgs_model = MLPClassifier(hidden_layer_sizes=(100,), solver='lbfgs', max_fun=5000)
lbfgs_model.fit(data, res)
loss_l = lbfgs_model.loss_
print(loss_l)
y = [loss_l, loss_l]
x = [1, len(loss_values)]
plt.plot(loss_values)
plt.plot(x, y)
plt.show()
print(adam_model.t_, adam_model.n_iter_)
print(lbfgs_model.t_, lbfgs_model.n_iter_)

# digits = load_digits()
# digits_data = np.array(digits['data']).reshape(-1, 64)
# digits_res = np.array(digits['target']).reshape(-1, 1)
# adam_model = MLPClassifier(hidden_layer_sizes=(100,), learning_rate="adaptive", solver='adam', max_iter=2000)
# adam_model.fit(digits_data, digits_res)
# loss_values = adam_model.loss_curve_
# lbfgs_model = MLPClassifier(hidden_layer_sizes=(100,), solver='lbfgs', max_fun=15000)
# lbfgs_model.fit(digits_data, digits_res)
# loss_l = lbfgs_model.loss_
# print(loss_l)
# y = [loss_l, loss_l]
# x = [1, len(loss_values)]
# plt.plot(loss_values)
# plt.plot(x, y)
# plt.show()
# print(adam_model.t_, adam_model.n_iter_)
# print(lbfgs_model.t_, lbfgs_model.n_iter_)