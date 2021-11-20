import os
import json
import torch



X_train = []
X_test = []
y_train = []
y_test = []
for id in range(1,6):
    path_train = os.path.join("data", "train", "mnist_train_" + "client" + str(id) + ".json")
    path_test = os.path.join("data", "test", "mnist_test_" + "client" + str(id) + ".json")
    data_train = {}
    data_test = {}

    with open(os.path.join(path_train), "r") as f_train:
        train = json.load(f_train)
        data_train.update(train['user_data'])
    with open(os.path.join(path_test), "r") as f_test:
        test = json.load(f_test)
        data_test.update(test['user_data'])

    X_T, y_T, X_t, y_t = data_train['0']['x'], data_train['0']['y'], data_test['0']['x'], data_test['0']['y']
    X_train += X_T
    X_test += X_t
    y_train += y_T
    y_test += y_t

y_T = [int(x) for x in y_T]
y_t = [int(x) for x in y_t]

X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
y_train = torch.Tensor(y_train).type(torch.int64)
X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
y_test = torch.Tensor(y_test).type(torch.int64)
train_samples, test_samples = len(y_train), len(y_test)
torch.save(torch.flatten(X_train, 1), 'X_train_MNIST.pt')
torch.save(y_train, 'y_train_MNIST.pt')
torch.save(torch.flatten(X_test, 1), 'X_test_MNIST.pt')
torch.save(y_test, 'y_test_MNIST.pt')
print(train_samples,test_samples)