import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.optim import LBFGS, Adam


X_train_MNIST = torch.load('X_train_MNIST.pt')
y_train_MNIST = torch.load('y_train_MNIST.pt')
X_test_MNIST = torch.load('X_test_MNIST.pt')
y_test_MNIST = torch.load('y_test_MNIST.pt')

X_train_iris = torch.load('X_train_iris.pt')
X_test_iris = torch.load('X_test_iris.pt')
y_train_iris = torch.load('y_train_iris.pt')
y_test_iris = torch.load('y_test_iris.pt')

train_samples_MNIST, test_samples_MNIST = len(y_train_MNIST), len(y_test_MNIST)
train_samples_iris, test_samples_iris = len(y_train_iris), len(y_test_iris)


class Net(nn.Module):
    def __init__(self, dataset_name, input_dim, output_dim, hidden_layer_sizes, criterion, sigmoid=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = hidden_layer_sizes
        self.iter = 0
        # The loss function could be MSE or BCELoss depending on the problem
        self.criterion = criterion

        # We leave the optimizer empty for now to assign flexibly
        self.optimiser = None

        hidden_layer_sizes = [input_dim] + hidden_layer_sizes
        last_layer = nn.Linear(hidden_layer_sizes[-1], self.output_dim)
        self.layers = \
            [nn.Sequential(nn.Linear(input_, output_), nn.ReLU())
             for input_, output_ in
             zip(hidden_layer_sizes, hidden_layer_sizes[1:])] + \
            [last_layer]

        # The output activation depends on the problem
        if sigmoid:
            self.layers = self.layers + [nn.Sigmoid()]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def train_(self, data_loader, epochs, validation_data=None):

        ### TensorBoard Writer Setup ###
        log_name = f"{'new-NN '+self.dataset_name}, {self.optimiser.__class__.__name__}"
        self.writer = SummaryWriter(log_dir=f"runs/{log_name}")
        print("To see tensorboard, run: tensorboard --logdir=runs/")

        for epoch in range(epochs):
            running_loss, average_loss = self._train_iteration(data_loader)

            test_acc = 0
            if validation_data is not None:
                y_hat = self(validation_data['X'])
                y_test_hat = validation_data['y']
                val_loss = self.criterion(input=y_hat, target=validation_data['y']).detach().numpy()

                test_acc += (torch.sum(torch.argmax(y_hat, dim=1) == y_test_hat) * 1. / y_test_hat.shape[0]).item()

                print('[%d] loss: %.3f | Average loss: %.3f | validation loss: %.3f | Accuracy: %.3f' %
                      (epoch + 1, running_loss, average_loss, val_loss, test_acc))
                self.writer.add_scalar('Average Loss', average_loss, global_step=epoch)
                self.writer.add_scalar('Validation Loss', val_loss, global_step=epoch)
                self.writer.add_scalar('Accuracy', test_acc, global_step=epoch)
            else:
                print('[%d] loss: %.3f | Average loss: %.3f' %
                      (epoch + 1, running_loss, average_loss))
                self.writer.add_scalar('Average Loss', average_loss, global_step=epoch)
        self.writer.close()

    def _train_iteration(self, data_loader):
        running_loss = 0.0
        i = 0
        for i, (X, y) in enumerate(data_loader):
            # Add the closure function to calculate the gradient.
            def closure():
                if torch.is_grad_enabled():
                    self.optimiser.zero_grad()
                output = self(X)
                loss = self.criterion(output, y)
                if loss.requires_grad:
                    loss.backward()
                return loss

            self.optimiser.step(closure)

            # calculate the loss again for monitoring
            output = self(X)
            loss = closure()
            running_loss += loss.item()

        average_loss = running_loss / (i + 1)

        return running_loss, average_loss


class ExperimentData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


criterion = nn.CrossEntropyLoss()

# MNIST -----------------------------------------------------------------------------------

# data_train_MNIST = ExperimentData(X_train_MNIST, y_train_MNIST)
# INPUT_SIZE_MNIST = X_train_MNIST.shape[1]
# OUTPUT_SIZE_MNIST = 10
# EPOCHS_MNIST = 40  # mind for overfitting
# data_loader_MNIST_Adam = DataLoader(data_train_MNIST, batch_size=128)
# data_loader_MNIST_LBFGS = DataLoader(data_train_MNIST, batch_size=X_train_MNIST.shape[0])
# # HIDDEN_LAYER_SIZE_MNIST = [392, 784, 392]
# HIDDEN_LAYER_SIZE_MNIST = [1000,1000]
#
# # Adam
# # print("Adam")
# # net_MNIST_Adam = Net("MNIST", INPUT_SIZE_MNIST, OUTPUT_SIZE_MNIST, HIDDEN_LAYER_SIZE_MNIST, criterion, sigmoid=False)
# # net_MNIST_Adam.optimiser = Adam(net_MNIST_Adam.parameters())
# # net_MNIST_Adam.train_(data_loader_MNIST_Adam, EPOCHS_MNIST, validation_data={"X": X_test_MNIST, "y": y_test_MNIST})
#
# # with SummaryWriter(comment='net_MNIST_Adam') as w:
# #     w.add_graph(net_MNIST_Adam, (X_train_MNIST,))
#
# # LBFGS
# print("LBFGS")
# net_MNIST_LBFGS = Net("MNIST", INPUT_SIZE_MNIST, OUTPUT_SIZE_MNIST, HIDDEN_LAYER_SIZE_MNIST, criterion, sigmoid=False)
# net_MNIST_LBFGS.optimiser = LBFGS(net_MNIST_LBFGS.parameters(), history_size=10, max_iter=4)
# net_MNIST_LBFGS.train_(data_loader_MNIST_LBFGS, EPOCHS_MNIST, validation_data={"X": X_test_MNIST, "y": y_test_MNIST})
# # with SummaryWriter(comment='net_MNIST_LBFGS') as w:
# #     w.add_graph(net_MNIST_LBFGS, (X_train_MNIST,))

# iris -----------------------------------------------------------------------------------

data_train_iris = ExperimentData(X_train_iris, y_train_iris)
INPUT_SIZE_iris = X_train_iris.shape[1]
OUTPUT_SIZE_iris = 3
EPOCHS_iris = 20
data_loader_iris_Adam = DataLoader(data_train_iris, batch_size=16)
data_loader_iris_IBFGS = DataLoader(data_train_iris, batch_size=X_train_iris.shape[0])
# HIDDEN_LAYER_SIZE_iris = []
HIDDEN_LAYER_SIZE_iris = [1000, 1000]
#
#
#
#
#
#
#
# LBFGS
print("LBFGS")
net_iris_LBFGS = Net("iris", INPUT_SIZE_iris, OUTPUT_SIZE_iris, HIDDEN_LAYER_SIZE_iris, criterion, sigmoid=False)
net_iris_LBFGS.optimiser = LBFGS(net_iris_LBFGS.parameters(), history_size=10, max_iter=4)
net_iris_LBFGS.train_(data_loader_iris_IBFGS, EPOCHS_iris, validation_data={"X": X_test_iris, "y": y_test_iris})
# with SummaryWriter(comment='net_iris_LBFGS') as w:
#     w.add_graph(net_iris_LBFGS, (X_train_iris,))

# Adam
print("Adam")
net_iris_Adam = Net("iris", INPUT_SIZE_iris, OUTPUT_SIZE_iris, HIDDEN_LAYER_SIZE_iris, criterion, sigmoid=False)
net_iris_Adam.optimiser = Adam(net_iris_Adam.parameters())
net_iris_Adam.train_(data_loader_iris_Adam, EPOCHS_iris, validation_data={"X": X_test_iris, "y": y_test_iris})

# with SummaryWriter(comment='net_iris_Adam') as w:
#     w.add_graph(net_iris_Adam, (X_train_iris,))