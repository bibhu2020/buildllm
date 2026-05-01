# Inference Instructions: Linear Regression (House)

**Model File**: `house_prediction_linear_regression.joblib`

### Summary
This model predicts house prices using a `ColumnTransformer` pipeline. It expects both numerical and categorical features.

### Inference Code
```python
import joblib
import pandas as pd

# 1. Load the model
model = joblib.load('src/ml/_model/house_prediction_linear_regression.joblib')

# 2. Input Data (Raw Dictionary)
raw_data = {
    'Living_Area': 2500,
    'Lot_Size': '0.5 ac', # Raw string with unit
    'Bedrooms': '3+1',    # Mixed string
    'Bathrooms': 2.5,
    'Neighborhood': 'North_Heights',
    'Property_Type': 'Single_Family',
    'Garage_Capacity': 2
}

# 3. Required Preprocessing (Must match training logic)
def preprocess(d):
    # Convert Lot Size (Acres to Sqft)
    d['Lot_Size'] = 0.5 * 43560 
    # Convert Bedrooms
    d['Bedrooms'] = 4
    return pd.DataFrame([d])

input_df = preprocess(raw_data)

# 4. Predict
price = model.predict(input_df)[0]
print(f"Predicted Price: ${price:,.2f}")
```
