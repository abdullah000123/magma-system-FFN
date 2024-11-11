import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# Define transformations for the dataset
transform = transforms.Compose([transforms.ToTensor()])
# Load the MNIST dataset
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
# Create data loaders for training and testing
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
# Initialization of weights and biases
no_input = 784   # MNIST images   784
no_l2 = 14       # Hidden layer 1
no_l3 = 14       # Hidden layer 2
no_out = 10      # Output layer for 10 classes

theta0 = np.random.rand(no_input, no_l2) * 0.001
bias0 = np.zeros((1, no_l2))
theta1 = np.random.rand(no_l2, no_l3) * 0.001
bias1 = np.zeros((1, no_l3))
theta2 = np.random.rand(no_l3, no_out) * 0.001
bias2 = np.zeros((1, no_out))

# Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Forward propagation
def forward_propagation(x):
    global a1, a2, a3
    z1 = np.dot(x, theta0) + bias0
    a1 = sigmoid(z1)
    z2 = np.dot(a1, theta1) + bias1
    a2 = sigmoid(z2)
    z3 = np.dot(a2, theta2) + bias2
    a3 = sigmoid(z3)
    return a3

# Backward propagation
def back_propagation(x, y, a3):
    global a1, a2
    # Calculate error at output layer
    error = y - a3
    s_bt = error * sigmoid_derivative(a3)  
    # Calculate errors for hidden layers
    er_s1 = np.dot(s_bt, theta2.T)
    s_b1 = er_s1 * sigmoid_derivative(a2)
    er_s2 = np.dot(s_b1, theta1.T)
    s_b2 = er_s2 * sigmoid_derivative(a1)   
    return s_bt, s_b1, s_b2

# Training function
def train(train_loader, lr, epochs):
    global theta1, theta2, bias1, bias2, theta0, bias0
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.view(-1, no_input).numpy()  # Flatten input images
            labels = np.eye(no_out)[labels]  # One-hot encode labels
            # Forward pass
            outputs = forward_propagation(inputs)
            # Calculate loss (Mean Squared Error)
            loss = np.mean(np.sum((labels - outputs)**2, axis=1))
            total_loss += loss
            # Backpropagation
            s_bt, s_b1, s_b2 = back_propagation(inputs, labels, outputs)
            # Update weights and biases
            theta2 += lr * np.dot(a2.T, s_bt)
            bias2 += lr * np.sum(s_bt, axis=0, keepdims=True)
            theta1 += lr * np.dot(a1.T, s_b1)
            bias1 += lr * np.sum(s_b1, axis=0, keepdims=True)
            theta0 += lr * np.dot(inputs.T, s_b2)
            bias0 += lr * np.sum(s_b2, axis=0, keepdims=True)
        
        # Print average loss per epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Set learning rate and epochsza
lr = 0.01
epochs = 200

# Train the model
train(train_loader, lr, epochs)
# Save weights and biases as separate .npy files
np.save('theta0.npy', theta0)
np.save('bias0.npy', bias0)
np.save('theta1.npy', theta1)
np.save('bias1.npy', bias1)
np.save('theta2.npy', theta2)
np.save('bias2.npy', bias2)
# save everything in one .npz file
np.savez('model_parameters.npz', theta0=theta0, bias0=bias0, theta1=theta1, bias1=bias1, theta2=theta2, bias2=bias2)
