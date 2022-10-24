# import the necessary packages
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d # Applies a 2D convolution over an input signal composed of several input planes
from torch.nn import Linear # fully connected layer(FC)
from torch.nn import MaxPool2d # To reduce the spatial dimensions of the input volume
from torch.nn import ReLU
from torch.nn import LogSoftmax # return the predicted probabilities of each class
from torch import flatten # flatten the output so we can apply FC layers

"""Initialize variables with layers to create CNN"""
class CNN(Module):
    def __init__(self,n_channels, classes):
        # call the parent constructor
        super(CNN, self).__init__()
        # initialize first layer conv to relu to pooling, input shape (1,28,28)
        self.layer1 = nn.Sequential(
            Conv2d(in_channels=n_channels, out_channels=25,   # convolutional layer with 25 filters of size 12x12x1, output shape (25,16,16)
                            kernel_size=12, stride=2),   # stride of 2 in both directions, no padding
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))  # max pooling layer with pool size 2x2, output shape (25,8,8)

        # initialize second layer
        self.layer2 = nn.Sequential(
            Conv2d(in_channels=25, out_channels=64,   # 64 filters of size 5x5x25, output shape (64,4,4)
                            kernel_size=5, stride=1, padding=2),   # stride of 1 in both directions, add padding if necessary
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)) #output shape  64*2*2

        # Add a fully connected layer to RELU layers with 1024 units
        self.fc1 = Linear(64 * 2 * 2, out_features=1024)
        self.relu3 = ReLU()

        # Add another FC with 1024 units input
        self.fc2 = Linear(in_features=1024, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)


# Define a forward function to connect my initialized variables and connected the layers
    def forward(self, x):
      x = self.layer1(x)
    # pass the output from the previous layer through the second
      x = self.layer2(x)
    # flatten the output from the previous layer and pass it to FC to RELU
      x = flatten(x, 1)
      x = self.fc1(x)
      x = self.relu3(x)
    # Add another FC to get 10 output units with softmax classifier
      x = self.fc2(x)
      output = self.logSoftmax(x)
    # return the output predictions
      return output

# import the necessary packages
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim import Adam  # Optimizer for model training
import matplotlib.pyplot as plt  # Library for data visualization

# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataloader():
# Load MNIST dataset
  train_data = datasets.MNIST(root = 'data', train = True, 
                transform = ToTensor(), download = True)
  test_data = datasets.MNIST(root = 'data', train = False, 
                transform = ToTensor())
  
  
  # Source of Data loader
  train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

  test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

  return train_loader, test_loader

#Training model
def train(train_loader, model, criterion, optimizer, epochs):
    # Train the model
  total_step = len(train_loader)
  for epoch in range(epochs):
    for iter, (x,y) in enumerate(train_loader): # x represnts image and y represents label
      (x,y) = (x.to(device), y.to(device))
        
      # Clear gradients
      optimizer.zero_grad()
        
      # Forward pass to get output/logits
      outputs = model(x)
        
      # Calculate Loss
      loss = criterion(outputs,y)
        
      # Perform backpropagation
      loss.backward()
        
      # Updating weights
      optimizer.step()

      # Keep track of the validation accuracy every 100 iterations    
      if (iter+1) % 100 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, iter + 1, total_step, loss.item()))



# Define iterable dataset for testing
def test(test_loader, model):
    # Test the model
    with torch.no_grad():
      # Calculate Accuracy         
      correct = 0
      total = 0
      # Iterate through test dataset
      for (x,y) in test_loader:
        (x,y) = (x.to(device), y.to(device))
                
        # Forward pass only to get output
        outputs = model(x)
                
        # calculate the number of correct predictions
        _, pred = torch.max(outputs.data, 1)
                
        # Total number of labels
        total += y.size(0)
        # Add the loss to the total training loss       
        correct += (pred == y).type(torch.float).sum().item()
      # Print Loss
      print('Accuracy of the network on the test images: {:.2f} %'.format(100 * correct / total))

"""Visualize the First Convolutional Filter"""
def visFilter(model):
    model_children=list(model.children())
    # Visualize 1st convolutional layer with shape 25*12*12
    filters=model_children[0][0].weight
    # Iterable filter from extracting all weights in CNN model
    filters=filters.detach().cpu()

    plt.figure(figsize=(12,12)) # Define figure size in image
    for idx, filter in enumerate (filters):
        # plot 25 filters each
        ax=plt.subplot(5,5,idx+1)

        ax.imshow(filter.squeeze(),cmap='gray') # Returns a tensor with all the dimensions of input with shape (25,1,12,12)
        ax.axis("off")
    # Save output image and show on view
    plt.savefig('./Conv1_22004074G.png')
    plt.show()

"""Running Model"""
# define training hyperparameters
learning_rate = 1e-4
batch_size = 50
n_iters = 3000

if __name__ == '__main__':
  # prepare dataset and create dataloader
  train_loader, test_loader = create_dataloader()

  # initialize model
  model = CNN(n_channels=1, classes=10)
  model.to(device)

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=learning_rate)

  # Train the model
  train(train_loader, model, criterion, optimizer, epochs=5)

  # Test the model
  test(test_loader, model)

  visFilter(model)
