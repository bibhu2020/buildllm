"""
PyTorch Tutorial Part 4: Datasets, DataLoaders, and GPU Support

Why do we need this?
1. Mini-Batch Training: In real-world scenarios, datasets are too large to fit into memory 
   all at once. We need to train on smaller chunks called "mini-batches".
2. `Dataset` and `DataLoader`: PyTorch provides these classes to easily handle data batching, 
   shuffling, and parallel loading.
3. GPU Acceleration: Deep learning models train much faster on GPUs. We need to explicitly 
   move our models and data to the GPU if one is available.

How to use this script:
This script builds upon Part 3 by adding:
1. Device configuration (`.to(device)` to support GPU if available).
2. A custom `TensorDataset` and `DataLoader` for mini-batching.
3. Updating the training loop to process data in mini-batches.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. Setup Device (Use GPU if available, else CPU)
# PyTorch allows you to dynamically check for a GPU and move your operations there.
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}\n")

# seed for reproducibility
torch.manual_seed(23)

# Generate synthetic data
# We increased the number of samples (N) to 1000 to better demonstrate the need for mini-batches.
N       = 1000  
D_in    = 1     # input feature dimension
D_out   = 1     # output feature dimension

# Objective is to find the weight & bias for the linear equation y = w*x + b
# True value of x is 2 and b is 3.
w_true = torch.tensor([[2.0]])
b_true = torch.tensor([[3.0]])

# Random Input Data
X = torch.randn(N, D_in)  
y_true = X @ w_true + b_true 
y_true = y_true + 0.1 * torch.randn(N, D_out) # adding noise

# Prepare training data
train_ratio = 0.8
split_index = int(N * train_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y_true[:split_index], y_true[split_index:]

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# 2. Create Dataset and DataLoader
# TensorDataset wraps our input and target tensors into a standard dataset object
train_dataset = TensorDataset(X_train, y_train)

# DataLoader automatically slices the dataset into mini-batches and shuffles the data
# Shuffling prevents the model from learning the order of the data.
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define a Custom Model by subclassing nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Instantiate the custom model and move it to the configured device (GPU/CPU)
model = LinearRegressionModel(input_dim=D_in, output_dim=D_out).to(device)

# Define Hyper-parameters
learning_rate = 0.05  # slightly reduced learning rate for mini-batch stability
epochs = 20           # Reduced total epochs since we now update weights multiple times per epoch!

# loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3. Mini-Batch Training Loop
for epoch in range(epochs):
    # Iterate over mini-batches provided by the DataLoader
    for batch_X, batch_y in train_loader:
        
        # CRITICAL: Move the mini-batch data to the same device as the model!
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        y_pred = model(batch_X) 
        
        # Calculate loss for the current mini-batch
        loss = loss_fn(y_pred, batch_y)

        # Backward pass & update
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad()

    # logging (we print at the end of every few epochs, note loss is just for the last batch)
    if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1:03d} | Loss: {loss.item():.6f} | w: {model.linear.weight.item():.4f} | b: {model.linear.bias.item():.4f}' )

# Evaluate the model with test data
with torch.no_grad():
    # Move test data to device for evaluation
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    y_test_pred = model(X_test)
    test_loss = loss_fn(y_test_pred, y_test)

# Compute Accuracy of the Model
ss_total = torch.sum((y_test - torch.mean(y_test)) ** 2)
ss_res = torch.sum((y_test - y_test_pred) ** 2)
r2_score = 1 - ss_res / ss_total

# Results
print("\n✅ Training complete!")
print(f"Learned weight (slope): {model.linear.weight.item():.4f}")
print(f"Learned bias (intercept): {model.linear.bias.item():.4f}")
print(f"Test MSE Loss: {test_loss.item():.6f}")
print(f"R² Score (accuracy): {r2_score.item():.4f}")
