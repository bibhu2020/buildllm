# Machine Learning Playground

This directory contains small samples and experiments for learning and validating concepts in Machine Learning, specifically focusing on PyTorch.

## PyTorch Tutorial Series

The following scripts form a progressive tutorial on PyTorch fundamentals, moving from manual gradient calculations to using high-level abstractions like Datasets and GPUs.

### 1. `01_tensors_and_autograd.py`: The Fundamentals (Tensors & Autograd)
Demonstrates the "under the hood" mechanics of PyTorch. It shows how to build a simple linear regression model entirely from scratch by manually defining parameters, tracking their gradients (`requires_grad=True`), and performing manual gradient descent using the Autograd engine.

### 2. `02_high_level_api_basics.py`: High-Level APIs (`nn.Module` & `optim`)
Shows how to simplify the process from Part 1 by introducing PyTorch's high-level abstractions. It replaces manual parameter definitions and weight updates with built-in modules (`nn.Linear`, `nn.MSELoss`) and optimizers (`torch.optim.SGD`).

### 3. `03_custom_nn_modules.py`: Custom Models (`nn.Module` Subclassing)
Introduces the standard PyTorch way to build complex neural networks. Instead of using a single built-in layer, it shows how to create custom models by subclassing `nn.Module`, defining the layers in `__init__`, and specifying the data flow in `forward()`.

### 4. `04_dataloaders_and_gpu.py`: Datasets, DataLoaders, & GPU Support
Brings the model closer to real-world deployment by introducing mini-batch training. It covers creating a `TensorDataset`, using a `DataLoader` for efficient batching and shuffling, and adding dynamic device management to train on a GPU (`.to(device)`) if available.

### 5. `05_end_to_end_regression.py`: Professional Regression Pipeline
The complete real-world regression pipeline using actual data (`student_performance.csv`). It implements data analysis with `pandas`, standard scaling with `scikit-learn`, a Linear Regression PyTorch model, and evaluates it using standard metrics like the $R^2$ Score and RMSE.

### 6. `06_logistic_regression_classification.py`: Professional Classification Pipeline
The complete real-world classification pipeline predicting heart disease risk (`framingham.csv`). It uses `BCEWithLogitsLoss` for numeric stability, handles missing data, evaluates the model using Accuracy/Precision/Recall/F1-Score, and demonstrates aggressively adjusting the decision threshold to reduce False Negatives in a medical context.