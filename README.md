# INFO3912Code

## Dataset preparation
1. The features and labels of iris dataset for training and testing are created by ```data_transformation_iris.py```, where the data is stored as .pt file and can be loaded as tensors.
2. The features and labels of MNIST dataset for training and testing are created by ```data_transformation_MNIST.py```, where the data is stored as .pt file and can be loaded as tensors.
3. Simply run ```python3 data_transformation_iris.py``` and ```python3 data_transformation_MNIST.py``` to re-generate them.

## Train and test
1. Run ```python3 Torch_NN``` to compare the performance of Adam and L-BFGS optimiers on iris and MNIST dataset using logistic regression and multi-layer nerural networks. The train loss, test loss and accuracy are outputed for each epoch.
2. Monitor the results on TensorBoard by runing ```tensorboard --logdir=runs/``` in terminal.