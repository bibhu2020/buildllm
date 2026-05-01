"""
PyTorch Tutorial Part 3: Custom Models with nn.Module Subclassing

Why do we need custom models?
While `pytorch2.py` showed how to use a single built-in layer (`nn.Linear`), 
real-world neural networks require combining many layers and defining custom 
logic for how data flows through them.

PyTorch provides a way to do this: subclassing `nn.Module`.
This is the standard way to build any model in PyTorch, from simple linear
regression to massive Transformer models.

How to use this script:
This script shows how to structure a PyTorch model properly:
1. Create a class that inherits from `nn.Module`.
2. In `__init__`, define the layers your model will use (e.g., `nn.Linear`).
3. In `forward()`, define how the input data passes through those layers.
4. The rest of the training loop remains exactly the same!
"""


import torch
import torch.nn as nn 

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


# Define a Custom Model by subclassing nn.Module
# This is the standard way to build neural networks in PyTorch.
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        # Always call the superclass __init__ first
        super().__init__()
        # Define the layers you want to use. 
        # Here we just use a single linear layer, but you could define many layers here.
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # Define how the data flows through the layers.
        # This is where the actual computation happens.
        return self.linear(x)

# Instantiate the custom model
model = LinearRegressionModel(input_dim=D_in, output_dim=D_out)



# Define Hyper-parameters
learning_rate = 0.1
epochs = 100

# loss function
loss_fn = nn.MSELoss()

# optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loops
for epoch in range(epochs):
    # Forward pass (model run) returns predicted data
    y_pred = model(X_train) 

    # Calculate loss (we will use "Mean Squared Error" as loss function here)
    loss = loss_fn(y_pred, y_train)

    # Backward pass: Automatically calculate gradients for all model parameters
    loss.backward() # Computes dloss/dw, dloss/db

    # Update weight and bias: The optimizer handles the parameter updates automatically
    optimizer.step() # Effectively does: w = w - learning_rate * w.grad

    # Zero out the gradients: PyTorch accumulates gradients by default, 
    # so we must reset them to zero before the next iteration
    optimizer.zero_grad()

    # logging
    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1:03d} | Loss: {loss.item():.6f} | w: {model.linear.weight.item():.4f} | b: {model.linear.bias.item():.4f}' )

# Evaluate the model with test data
with torch.no_grad():
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