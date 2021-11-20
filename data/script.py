import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.optim import LBFGS, Adam
# class Net(nn.Module):
#     def __init__(self, input_dim, hidden_layer_sizes, loss, sigmoid=False):
#         super(Net, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_layer_sizes = hidden_layer_sizes
#         self.iter = 0
#         self.loss_func = loss
#
#
#
#
#
#         self.fc1 = nn.Linear(784, 10)
#
#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         print(x.shape)
#         x = self.fc1(x)
#         return x

# def train(epochs, model, loss, trainloader, optimizer):
#     LOSS = []
#     model.train()
#     for epoch in range(1, epochs + 1):
#         model.train()
#         for batch_idx, (X, y) in enumerate(trainloader):
#             optimizer.zero_grad()
#             print(X.shape)
#             output = model(X)
#             loss_val = loss(output, y)
#             # print("loss:", loss.numpy())
#             # loss_per_epoch += loss
#             loss_val.backward()
#             optimizer.step()
#         LOSS.append(loss_val.data)
#     return LOSS
#
# def test(model, testloader):
#     model.eval()
#     test_acc = 0
#     for x, y in testloader:
#         output = model(x)
#         test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
#         print("Accuracy of client is: ", test_acc)
#     return test_acc

# model1 = MCLR()
# batch_size = 200
# total_train_samples = 0
# num_glob_iters = 100
# train_data = [(x, y) for x, y in zip(X_train, y_train)]
# test_data = [(x, y) for x, y in zip(X_test, y_test)]
# trainloader = DataLoader(train_data, batch_size)
# testloader = DataLoader(test_data, test_samples)
# criterion = nn.CrossEntropyLoss()
#
# loss_value = train(100, model1, criterion, trainloader, Adam(model1.parameters()))
# plt.figure(1,figsize=(5, 5))
# plt.plot(loss_value, label="Adam", linewidth  = 1)
# plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
# plt.ylabel('Training Loss')
# plt.xlabel('Global rounds')
# plt.show()
#
# print(test(model1, testloader))