# Inference Instructions: Logistic Regression (Email)

**Model File**: `email_classification_logistic_regression.joblib`

### Summary
This model is a Scikit-Learn Pipeline that includes both a `TfidfVectorizer` and a `LogisticRegression` estimator. It is designed to handle "dirty" email data with HTML artifacts.

### Inference Code
```python
import joblib
import re

# 1. Load the model
model = joblib.load('src/ml/_model/email_classification_logistic_regression.joblib')

# 2. Input Data (Raw)
raw_email = "<div>CONGRATULATIONS! You won a FREE prize. Click here.</div>"

# 3. Required Preprocessing
def clean(text):
    text = re.sub('<.*?>', '', text) # Strip HTML
    return text.strip().lower()

processed_text = [clean(raw_email)]

# 4. Predict
prediction = model.predict(processed_text)
probability = model.predict_proba(processed_text)[0][1]

print(f"Spam Prediction: {prediction[0]}")
print(f"Spam Probability: {probability:.4f}")
```
