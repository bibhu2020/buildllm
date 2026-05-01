"""
PyTorch Tutorial Part 1: Tensors, Autograd, and Manual Gradient Descent

Why do we need PyTorch?
1. Tensor Computation: Similar to NumPy, but PyTorch Tensors can run on GPUs to accelerate computing.
2. Automatic Differentiation (Autograd): It automatically calculates gradients (derivatives), 
   which is the math required to train machine learning models via backpropagation.

How to use this script:
This script demonstrates the "under the hood" mechanics of PyTorch. We will build a simple
linear regression model (y = wx + b) entirely from scratch. We manually define parameters, 
tell PyTorch to track their gradients (`requires_grad=True`), and manually update them.
"""

import torch

# seed for reproducibility
torch.manual_seed(23)

# Generate synthetic data
N       = 50    # no of samples
D_in    = 1     # input feature dimension
D_out   = 1     # output feature dimension

# Objective is to find the weight & bias for the linear equation y = w*x + b
# True value of x is 2 and b is 3.
w_true = torch.tensor([[2.0]])
b_true = torch.tensor([[3.0]])

# Random Input Data
X = torch.randn(N, D_in)  # It creates 1D tensor with 50 random input x values
#print(X) # tensor([[-0.9012],[ 0.5656],..............,[-0.6442]])

# True Value of y should be (y_true = w_true * X + b_true)
# We will add some noise to it. Otherwise, y_true is just perfect, and we need no training
y_true = X @ w_true + b_true 
y_true = y_true + 0.1 * torch.randn(N, D_out) # adding noise


# Prepare training data
train_ratio = 0.8
split_index = int(N * train_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y_true[:split_index], y_true[split_index:]

print("Train size:", X_train.shape[0], " | Test size:", X_test.shape[0])

# At this point, we do not know the true values of w and b.
# We initialize them with random values. 
# CRITICAL: We set requires_grad=True so PyTorch's Autograd engine tracks all operations 
# on these tensors. This allows PyTorch to automatically compute gradients later.
w = torch.randn((D_in, D_out), requires_grad=True)
b = torch.randn((1, D_out), requires_grad=True)
print(f"Initial weights: {w}, Initial bias: {b}")

# model forward pass
def forward(x):
    return x @ w + b


# Define Hyper-parameters
learning_rate = 0.1
epochs = 100

# Training loops
for epoch in range(epochs):
    # Forward pass (model run) returns predicted data
    y_pred = forward(X_train)

    # Calculate loss (we will use "Mean Squared Error" as loss function here)
    loss = torch.mean((y_pred - y_train) ** 2)

    # Backward pass: This is where PyTorch's Autograd shines!
    # Calling .backward() automatically computes the derivative of the loss 
    # with respect to all tensors that have requires_grad=True (our w and b).
    loss.backward() 

    # Update weight and bias
    with torch.no_grad():
        w -= learning_rate * w.grad 
        b -= learning_rate * b.grad 

    # Reset gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()

    # logging
    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1:03d} | Loss: {loss.item():.6f} | w: {w.item():.4f} | b: {b.item():.4f}' )

# Evaluate the model with test data
with torch.no_grad():
    y_test_pred = X_test @ w + b
    test_loss = torch.mean((y_test_pred - y_test) ** 2)

# Compute Accuracy of the Model
ss_total = torch.sum((y_test - torch.mean(y_test)) ** 2)
ss_res = torch.sum((y_test - y_test_pred) ** 2)
r2_score = 1 - ss_res / ss_total

# Results
print("\n✅ Training complete!")
print(f"Learned weight (slope): {w.item():.4f}")
print(f"Learned bias (intercept): {b.item():.4f}")
print(f"Test MSE Loss: {test_loss.item():.6f}")
print(f"R² Score (accuracy): {r2_score.item():.4f}")