# Codec Technologies AI Internship - Project Portfolio

This repository contains two completed machine learning and natural language processing mini-projects as part of the 1-Month Artificial Intelligence Internship program at Codec Technologies.

---

## Project 1: Stock Price Predictor

### Objective
To develop a predictive model that estimates future stock prices using historical market data.

### Code Implementation (`stock_predictor.py`)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample Dataset Simulation for Demonstration
np.random.seed(42)
days = np.arange(1, 101).reshape(-1, 1)
prices = 50 + 1.5 * days.ravel() + np.random.normal(0, 5, 100)

# Data Preparation
X = days
y = prices

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions & Evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"Model Coefficient: {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
