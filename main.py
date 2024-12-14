import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# function to split data into training and validation
def split_data(data, train_ratio=0.8):
    shuffle_data = np.random.permutation(data.shape[0]) # shuffle randomly
    train_size = int(data.shape[0] * train_ratio) # the size of training
    train_data = shuffle_data[:train_size] # from 0 to train_size-1
    val_data = shuffle_data[train_size:] # from train_size to the end
    return data.iloc[train_data], data.iloc[val_data] # return the rows of the data

# split train data into 80% train and 20% validation
train_data, val_data = split_data(train_data, train_ratio=0.8)

# process train and validation data
x_train = train_data.iloc[:, 1:].values / 255.0  # normalize (0, 255) -> (0, 1)
y_train = train_data.iloc[:, 0].values # labels
x_val = val_data.iloc[:, 1:].values / 255.0 # features
y_val = val_data.iloc[:, 0].values # labels

x_test = test_data.iloc[:, 0:].values / 255.0

# one-hot encoding
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

num_classes = 10
y_train_one_hot = one_hot(y_train, num_classes)
y_val_one_hot = one_hot(y_val, num_classes)

# initialize parameters
input_size = 784
hidden_size = 128
output_size = 10

np.random.seed(0)

# initialize w, b
W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))

# ReLU
def relu(n):
    return np.maximum(0, n) # f(x) = max(0, x)

# softmax
def softmax(n):
    exp_n = np.exp(n - np.max(n, axis=0, keepdims=True)) # subtract max for stability and wont change the ans
    return exp_n / np.sum(exp_n, axis=0, keepdims=True) # normalize

# forward propagation
def forward(x, W1, b1, W2, b2):
    x = x.T # transpose the matrix so that it can fit the shape
    n1 = np.matmul(W1, x) + b1 # n = wx + b
    a1 = relu(n1)
    n2 = np.matmul(W2, a1) + b2
    a2 = softmax(n2)
    return n1, a1, n2, a2

# loss function
def cross_entropy_loss(y_pred, y):
    batch_size = y.shape[0]
    loss = -np.sum(y * np.log(y_pred.T)) / batch_size  # divide by the batch size to normalize the loss
    return loss

# backward propagation
def backward(x, y, n1, a1, n2, a2, W1, W2):
    batch_size = x.shape[0]
    dn2 = a2.T - y # Δ = a - y
    dW2 = np.matmul(dn2.T, a1.T) / batch_size 
    db2 = np.sum(dn2, axis=0, keepdims=True).T / batch_size
    da1 = np.matmul(W2.T, dn2.T)
    dn1 = da1 * relu(n1) # ReLU
    dW1 = np.matmul(dn1, x) / batch_size
    db1 = np.sum(dn1, axis=1, keepdims=True) / batch_size
    return dW1, db1, dW2, db2

def evaluate(x, y):
    _, _, _, a2 = forward(x, W1, b1, W2, b2)
    y_pred = np.argmax(a2.T, axis=1) # the maximum value corresponds to the predicted class
    y_true = np.argmax(y, axis=1)  # the maximum value corresponds to the predicted class
    correct = np.sum(y_pred == y_true) # the number of correct predictions
    accuracy = correct / x.shape[0]
    return accuracy

# hyperparameters
learning_rate = 0.1
epochs = 500
mini_batch_size = 256

# restore the losses
train_losses = []
val_losses = []

for epoch in range(epochs):
    random_idx = np.random.permutation(x_train.shape[0]) # shuffle randomly to prevent any bias
    x_train_shuffle = x_train[random_idx]
    y_train_shuffle = y_train_one_hot[random_idx]
    x_val_shuffle = x_train[random_idx]
    y_val_shuffle = y_train_one_hot[random_idx]

    train_loss = 0
    for i in range(0, x_train.shape[0], mini_batch_size):
        x_batch = x_train_shuffle[i:i + mini_batch_size]
        y_batch = y_train_shuffle[i:i + mini_batch_size]

        # forward propagation
        n1, a1, n2, a2 = forward(x_batch, W1, b1, W2, b2)
        
        # loss
        batch_loss = cross_entropy_loss(a2, y_batch)
        train_loss += batch_loss

        # backward propagation
        dW1, db1, dW2, db2 = backward(x_batch, y_batch, n1, a1, n2, a2, W1, W2)

        # update w, b
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    train_loss /= (x_train.shape[0] // mini_batch_size)
    train_losses.append(train_loss)

    # validate the last 20% data
    _, _, _, a2_val = forward(x_val, W1, b1, W2, b2) # only a2 is needed for the validation
    val_loss = cross_entropy_loss(a2_val, y_val_one_hot)
    val_losses.append(val_loss)

    print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    if train_loss < 0.05 or val_loss < 0.05:
        break

# evaluate to get the accuracy
train_acc = evaluate(x_train, y_train_one_hot) * 100
val_acc = evaluate(x_val, y_val_one_hot) * 100
test_preds = np.argmax(forward(x_test, W1, b1, W2, b2)[3].T, axis=1) # the maximum value corresponds to the predicted class

#  text_output.txt
with open('text_output.txt', 'w') as f:
    for pred in test_preds:
        f.write(f"{pred}\n")

# print result
print(f"Layers: 3")
print(f"Layer0: {input_size}")
print(f"Layer1: {hidden_size}")
print(f"Layer2: {output_size}")
print(f"Max Epoch: {epoch}")
print(f"Learning Rate: {learning_rate}")
print(f"Mini Batch size: {mini_batch_size}")
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Validation Accuracy: {val_acc:.2f}%")
        
# convert train_losses and val_losses to numpy
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

# plot the loss curves
plt.figure(figsize=(8, 6))
plt.plot(range(1, train_losses.shape[0] + 1), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, val_losses.shape[0] + 1), val_losses, marker='x', label='Validation Loss')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('output.png')
plt.show()

