# Model Inference & Deployment Guide

This directory contains the persisted artifacts for the Email, House, and Tesla machine learning suites. Below are instructions for loading and using these models for real-time inference.

---

## 1. Classical Models (Scikit-Learn / `.joblib`)
These models are saved as full pipelines, meaning the vectorizers and scalers are embedded within the file.

### Models:
- `email_classification_logistic_regression.joblib`
- `house_prediction_random_forest.joblib`
- `tesla_trading_random_forest.joblib`
- *(And all other .joblib files)*

### How to use:
```python
import joblib

# 1. Load the pipeline
pipeline = joblib.load('src/ml/_model/email_classification_logistic_regression.joblib')

# 2. Preprocess your raw input (e.g., strip HTML)
raw_text = "<div>FREE WINNER!</div>"
clean_text = [raw_text.replace('<div>', '').replace('</div>', '')]

# 3. Predict
prediction = pipeline.predict(clean_text)
print(f"Is Spam: {prediction[0]}")
```

---

## 2. Deep Learning Models (PyTorch / `.pth`)
These models require the original class definition to be present in your script before loading the weights.

### Models:
- `email_classification_rnn.pth`
- `house_prediction_tab_transformer.pth`
- `tesla_trading_lstm.pth`

### How to use:
```python
import torch
from your_model_defs import SpamRNN # Import the class from the notebook/script

# 1. Initialize architecture
model = SpamRNN(vocab_size=5001, embed_dim=32, hidden_dim=64)

# 2. Load weights
model.load_state_dict(torch.load('src/ml/_model/email_classification_rnn.pth'))
model.eval()

# 3. Tokenize and predict
# (Ensure you use the same vocab mapping saved in the .pickle files)
```

---

## 3. Important Preprocessing Notes
For successful inference, you MUST apply the same logic used during training:

| Project | Required Cleanup Action |
| :--- | :--- |
| **Email** | Strip all HTML tags (`<.*?>`) and lowercase text. |
| **House** | Normalize `Lot_Size` to Square Feet and map Condition strings to 1-10. |
| **Tesla** | Back-adjust prices using the `Split_Factor` if analyzing historical trends. |

---

## 4. Troubleshooting
- **Casing Errors**: Ensure all input dataframes use lowercase column names (`snake_case`).
- **Version Mismatch**: These models were built with `scikit-learn==1.3.x` and `torch==2.0.x`. Ensure your environment matches to avoid loading errors.
