#This is a Machine learning model that predicts the Fashion MNIST dataset
#This program aims to define a Convolutional Neural Network
# and compare it's accuracy and time with Multilayer Perceptron (MLP)
#Coded by Mahmoud-K-Ismail

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, \
    Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Flatten
from torch.optim import Adam, SGD
import torch.nn.functional as F
import time
from torch.nn.functional import nll_loss, cross_entropy
import torch.optim as optim

## Loading and preparing the FashionMNIST dataset"""

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Visualize some images of the FashionMNIST dataset
# Size of training data
print(train_data.data.shape)

# Size of testing data
print(test_data.data.shape)

labels_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 4, 4
for i in range(1, cols * rows + 1):
    sample_idx = i
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Initializing a dataloader with batch_size of 32
BATCH_SIZE = 32
trainloader = DataLoader(train_data,batch_size = BATCH_SIZE , shuffle = True)
testloader = DataLoader(test_data,batch_size = BATCH_SIZE , shuffle = False)


#Defining the Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cnn_layers = Sequential(

            #first we fix a kernel
            #stride is how much you move every step
            #adds a padding so that our kernel multiplies patches on the border with more detail
            #4 is the output channel so we have 4 kernels being learned

            # Defining a 2D convolution layer with a kernel of size 3; padding 1, and stride 1.
            nn.Conv2d(in_channels = 1 , out_channels = 4, padding = 1, kernel_size = 3, stride = 1),
            nn.ReLU(),
            # Defining another 2D convolution layer
            nn.Conv2d(in_channels = 4 , out_channels = 8, padding = 1, kernel_size = 3, stride = 1),
            nn.ReLU()

        )
        # Define one linear layer
        self.linear_layers = Sequential(
            Flatten(),  # Flatten the output from the CNN layers
            nn.Linear(8 * 28 * 28, 10),  # 4 channels * 28x28 image size from convolutions
        )

    # Defining the forward pass
    def forward(self, x):
        z = self.cnn_layers(x)
        z = self.linear_layers(z) # to do
        return z



##Training my CNN on the training set of FashionMNIST.

start_time = time.time()
convnet = ConvNet()
print(convnet.parameters)
# Optimizer
epochs = 5
learning_rate = 1e-4
optimizer = Adam(convnet.parameters(), lr=learning_rate)

# Choice of the loss
criterion = CrossEntropyLoss() # nll_loss

losses_cnn = []
for t in range(epochs):
    for i, data in enumerate(trainloader):
        inputs, labels = data

        # set optimizer to zero grad to remove previous epoch gradients
        optimizer.zero_grad()

        # Evaluate the loss
        inputs= inputs.view(32, 1, 28, 28)  # Assuming each image is 28x28 and has 1 channel (grayscale)
        outputs = convnet.forward(inputs)# to do
        loss = criterion(outputs,labels) # to do

        # backward propagation
        # to do
        loss.backward()
        # One optimization step
        optimizer.step()
        # to do

        losses_cnn.append(loss.item())


        if not i % 2000:
            print(t, i, loss.item())

end_time = time.time()
elapsed_time_CNN = end_time - start_time


## checking the accuracy of the CNN model on the testing set of FashionMNIST
size_test = test_data.data.shape[0]

correct = 0
for data in testloader:
    inputs, labels = data
    inputs= inputs.view(-1, 1, 28, 28)
    outputs = convnet(inputs)
    loss = criterion(outputs,labels)
    predicted = torch.argmax(outputs, dim=1)
    correct += (predicted == labels).sum()

final_accuracy_CNN =  correct/size_test

############################ Initializing Training a Multilayer Perceptron############

# number of features (len of X cols)
input_dim = 28 * 28

# number of hidden layers
hidden_dim = 256

# number of classes (unique of y)
output_dim = 10

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear (hidden_dim,output_dim)

    def forward(self, x):
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred

start_time = time.time()
mlp = MLP()
print(mlp.parameters)

# Optimizer
epochs = 5
learning_rate = 1e-6
optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

# Choice of the loss
criterion = cross_entropy # nll_loss

losses = []
for t in range(epochs):
    for i, data in enumerate(trainloader):
        inputs, labels = data

        # set optimizer to zero grad to remove previous epoch gradients
        optimizer.zero_grad()

        # Evaluate the loss
        inputs = inputs.view(-1, input_dim)
        outputs = mlp(inputs)  #### not sure what input shoud I pass
        # loss = nll_loss(outputs, labels)
        loss = criterion(outputs,labels)

        # backward propagation
        loss.backward()
        # One optimization step
        optimizer.step()
        losses.append(loss.item())

        if not i % 2000:
            print(t, i, loss.item())
#calculating time
end_time = time.time()
elapsed_time_MLP = end_time - start_time

#ploting the loss vs Iterations in MLP
plt.plot(losses, label = "MLP Loss")
plt.plot(losses_cnn, label="CNN Loss")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss over iterations")
plt.show()


## checking the accuracy of the MLP model on the testing set of FashionMNIST
size_test = test_data.data.shape[0]

correct = 0
for data in testloader:
    inputs, labels = data
    inputs = inputs.view(-1, input_dim)
    outputs = mlp(inputs)
    loss = criterion(outputs,labels)
    predicted = torch.argmax(outputs, dim=1)
    correct += (predicted == labels).sum()

final_accuracy_MLP =  correct/size_test


####################### Printing Conclusions ############################
#CNN accuracy
print(f"The CNN accuracy is: {final_accuracy_CNN}")

#MLP accuracy
print(f"The MLP accuracy is: {final_accuracy_MLP}")

# Calculate the elapsed time
print("Elapsed time For Conv neural network is :", elapsed_time_CNN, "seconds")

# Calculate the elapsed time
print("Elapsed time for MLP is :", elapsed_time_MLP, "seconds")




"""So in terms of time, multilinear perception is faster than the convolutional neural networks, however, in terms of accuracy the CNN surpases the MLP by far."""
