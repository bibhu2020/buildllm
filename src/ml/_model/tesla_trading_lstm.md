# Inference Instructions: LSTM (Tesla Stock)

**Model File**: `tesla_trading_lstm.pth`

### Summary
This is a PyTorch-based Long Short-Term Memory network designed for time-series price forecasting. It requires a 30-day lookback window.

### Inference Code
```python
import torch
import numpy as np
# from your_model_file import StockLSTM

# 1. Define Architecture
model = StockLSTM(input_dim=3, hidden_dim=64)

# 2. Load Weights
model.load_state_dict(torch.load('src/ml/_model/tesla_trading_lstm.pth'))
model.eval()

# 3. Prepare Input (30-day sequence of [Close, Volume, RSI])
# Sequence shape must be [1, 30, 3]
sample_sequence = np.random.rand(1, 30, 3) 
input_tensor = torch.tensor(sample_sequence, dtype=torch.float32)

# 4. Predict
with torch.no_grad():
    prediction_scaled = model(input_tensor).item()

print(f"Predicted Next Day Price (Scaled): {prediction_scaled:.4f}")
```
