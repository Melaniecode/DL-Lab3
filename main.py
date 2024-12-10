import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt   
import random

# load data 
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# process data
x_train = train_data.iloc[:, 1:].values / 255.0 # (0, 255) -> (0, 1)
y_train = train_data.iloc[:, 0].values # use .values to return a Numpy representation of the DataFrame
x_test = test_data.iloc[:, 0:].values / 255.0 # (0, 255) -> (0, 1)
y_test = test_data.iloc[:, 0].values # use .values to return a Numpy representation of the DataFrame.

# one-hot encoding
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

W1 = np.random.randn(hidden_size, input_size) # weight for the first layer (128 * 784)
b1 = np.zeros((hidden_size, 1)) # bias for the first layer (128 * 1)

W2 = np.random.randn(output_size, hidden_size) # weight for the second layer (10 * 128)
b2 = np.zeros((output_size, 1)) # bias for the second layer (10 * 1)

# forward propagation
def forward(x, W1, b1, W2, b2):
    x = x.T # transpose to match the shape
    n1 = np.matmul(W1, x) + b1
    a1 = relu(n1)
    n2 = np.matmul(W2, a1) + b2
    a2 = softmax(n2)
    return n1, a1, n2, a2

# activation function
def relu(n):
    return np.maximum(0, n)

def softmax(n):
    exp_n = np.exp(n - np.max(n, axis=0, keepdims=True))  #  substract max(n) to prevent overflow
    return exp_n / np.sum(exp_n, axis=0, keepdims=True)

# loss function
def cross_entropy_loss(y_pred, y):
    batch_size = y.shape[0] # prevent the gradient from becoming too large
    loss = -np.sum(y * np.log(y_pred.T + 1e-8)) / batch_size  # add (1e-8) to prevent underflow or log(0)
    return loss

# backward propagation
def backward(x, y, n1, a1, n2, a2, W1, W2):
    batch_size = x.shape[0] # prevent the gradient from becoming too large
    
    dn2 = a2.T - y
    dW2 = np.matmul(dn2.T, a1.T) / batch_size
    db2 = np.sum(dn2, axis=0, keepdims=True).T / batch_size

    da1 = np.matmul(W2.T, dn2.T)
    dn1 = da1 * (n1 > 0) # relu
    dW1 = np.matmul(dn1, x) / batch_size
    db1 = np.sum(dn1, axis=1, keepdims=True) / batch_size
    
    return dW1, db1, dW2, db2

# hyperparameters
learning_rate = 0.1
epochs = 50
mini_batch_size = 256

# lists to store losses
train_losses = [] 
test_losses = []

# evaluation function
def evaluate(x, y):
    _, _, _, a2 = forward(x, W1, b1, W2, b2)
    predictions = np.argmax(a2.T, axis=1)
    accuracy = np.mean(predictions == np.argmax(y, axis=1))
    return accuracy

for epoch in range(epochs):
    # Shuffle training data
    random_idx = np.random.permutation(x_train.shape[0])  # ensures that each mini-batch has a more balanced representation of all classes
    x_train_shuffle = x_train[random_idx]
    y_train_shuffle = y_train_one_hot[random_idx]
    
    epoch_loss = 0
    
    for i in range(0, x_train.shape[0], mini_batch_size):
        x_batch = x_train_shuffle[i:i + mini_batch_size]
        y_batch = y_train_shuffle[i:i + mini_batch_size]
        
        # forward propagation
        n1, a1, n2, a2 = forward(x_batch, W1, b1, W2, b2)
        
        # loss
        batch_loss = cross_entropy_loss(a2, y_batch)
        epoch_loss += batch_loss
        
        # backward propagation
        dW1, db1, dW2, db2 = backward(x_batch, y_batch, n1, a1, n2, a2, W1, W2)
        
        # update weights and biases
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
    
    epoch_loss /= (x_train.shape[0] // mini_batch_size) # reduce the variation from mini-batch training
    train_losses.append(epoch_loss)
    
    _, _, _, a2_test = forward(x_test, W1, b1, W2, b2)
    test_loss = cross_entropy_loss(a2_test, y_test_one_hot)
    test_losses.append(test_loss)
    
    if epoch_loss < 0.1:
        break

# plot training and test loss curves
plt.figure(figsize=(8, 6))
plt.plot(range(1, epoch + 2), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, epoch + 2), test_losses, marker='x', label='Test Loss')
plt.title('Training and Test Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# evaluate accuracy
train_acc = evaluate(x_train, y_train_one_hot) * 100
test_acc = evaluate(x_test, y_test_one_hot) * 100

# print result
print(f"Layers: 3")
print(f"Layer0: {input_size}")
print(f"Layer1: {hidden_size}")
print(f"Layer2: {output_size}")
print(f"Max Epoch: {epoch}")
print(f"Learning Rate: {learning_rate}")
print(f"Mini Batch size: {mini_batch_size}")
print(f"Train accuracy: {train_acc:.2f}%")
print(f"Test accuracy: {test_acc:.2f}%") # noooooooo still overfitting!!!

# testing
_, _, _, a2_test = forward(x_test, W1, b1, W2, b2)

predictions = np.argmax(a2_test.T, axis=1)

#  text_output.txt
with open('text_output.txt', 'w') as f:
    for pred in predictions:
        f.write(f"{pred}\n")