import numpy as np
import matplotlib.pyplot as plt
import torch
plt.style.use('ggplot')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_scaled, y, test_size=0.25, random_state=2)

X_train_iris = torch.from_numpy(np.array(X_train_iris).reshape(-1, 4)).type(torch.float32)
X_test_iris = torch.from_numpy(np.array(X_test_iris).reshape(-1, 4)).type(torch.float32)
y_train_iris = torch.from_numpy(np.array(y_train_iris).reshape(-1, 1)).type(torch.int64).squeeze(1)
y_test_iris = torch.from_numpy(np.array(y_test_iris).reshape(-1, 1)).type(torch.int64).squeeze(1)

torch.save(X_train_iris, 'X_train_iris.pt')
torch.save(X_test_iris, 'X_test_iris.pt')
torch.save(y_train_iris, 'y_train_iris.pt')
torch.save(y_test_iris, 'y_test_iris.pt')
print(len(X_train_iris), len(X_test_iris))