# Import libraries
import pandas as pd
from pycaret.classification import setup, compare_models

# Load dataset
from sklearn.datasets import load_iris
data = load_iris(as_frame=True)
df = data['data']
df['target'] = data['target']

# Step 1: Initialize the PyCaret environment
clf = setup(data=df, target='target', silent=True, session_id=42)

# Step 2: Compare models and choose the best one
best_model = compare_models()

# Step 3: Print the best model
print("Best Model:")
print(best_model)
