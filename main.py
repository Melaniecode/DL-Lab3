import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt   
import random

# load data 
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# process data
x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
x_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# One-Hot Encoding
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

num_classes = 10

y_train_one_hot = one_hot(y_train, num_classes)
y_test_one_hot = one_hot(y_test, num_classes)

# initialize parameters
input_size = 784  # (28 * 28)
hidden_size = 128  
output_size = 10

np.random.seed(0)

# TODO: 檢查矩陣大小
W1 = np.random.randn(input_size, hidden_size) # weight for the first layer
b1 = np.zeros((1, hidden_size)) # bias for the first layer

W2 = np.random.randn(hidden_size, output_size) # weight for the second layer
b2 = np.zeros((1, output_size)) # bias for the second layer

# TODO: 應為（w, x.T)
# forward propagation
def forward(x, W1, b1, W2, b2):
    n1 = np.matmul(x, W1) + b1
    a1 = relu(n1)
    n2 = np.matmul(a1, W2) + b2
    a2 = softmax(n2)
    return n1, a1, n2, a2

# activation function
def relu(n):
    return np.maximum(0, n)

def softmax(n):
    exp_n = np.exp(n - np.max(n, axis=1, keepdims=True))  #  prevent overflow
    return exp_n / np.sum(exp_n, axis=1, keepdims=True)

# loss funtion
def cross_entropy_loss(y_pred, y):
    

# backward propagation
# def backward(): 

# hyperparameters
learning_rate = 0.1
epochs = 10
mini_batch_size = 64

train_losses = [] 
test_losses = []

for epoch in range(epochs):
    
    # shuffling
    perm = np.random.permutation(x_train.shape[0])
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train_one_hot[perm]
    
    epoch_loss = 0
    
    for i in range(0, x_train.shape[0], mini_batch_size):
        x_batch = x_train_shuffled[i:i+mini_batch_size]
        y_batch = y_train_shuffled[i:i+mini_batch_size]
        
        # forward propagation
        n1, a1, n2, a2 = forward(x_batch, W1, b1, W2, b2)
        
        # TODO: loss
        batch_loss = cross_entropy_loss(a2, y_batch)
        
        # TODO: backward propagation
        # TODO: 更新 w, b
        
# TODO: 模型正確度
    
# TODO: 畫圖
# TODO: print 結果到 test_output.txt