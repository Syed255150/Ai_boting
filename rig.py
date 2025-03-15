import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# Load Real-World Dataset (California Housing Prices)
data = fetch_california_housing()
X = data.data[:, :3]  # Use first 3 features correctly
y = data.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train & Test Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to Train & Evaluate a Model
def evaluate_model(model, X_train, X_test, y_train, y_test, name="Model"):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"{name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    return test_r2  # Return test R² score

# Train & Evaluate Linear Regression
linear_model = LinearRegression()
linear_r2 = evaluate_model(linear_model, X_train, X_test, y_train, y_test, "Linear Regression")

# Polynomial Regression (Degree 2 & 3)
poly_r2_scores = []
degrees = range(2, 4)  # Only using degree 2 & 3 to avoid overfitting

for degree in degrees:
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_r2 = evaluate_model(poly_model, X_train, X_test, y_train, y_test, f"Polynomial Degree {degree}")
    poly_r2_scores.append(poly_r2)

# Ridge & Lasso Regression with optimized alpha values
ridge_model = Ridge(alpha=0.5)
ridge_r2 = evaluate_model(ridge_model, X_train, X_test, y_train, y_test, "Ridge Regression")

lasso_model = Lasso(alpha=0.05)
lasso_r2 = evaluate_model(lasso_model, X_train, X_test, y_train, y_test, "Lasso Regression")

# Cross-Validation Scores for Linear Regression
cv_scores = cross_val_score(linear_model, X_scaled, y, cv=5, scoring='r2')
print(f"Linear Regression - Cross-Validation R² Scores: {cv_scores}")
print(f"Linear Regression - Mean CV R² Score: {np.mean(cv_scores):.4f}")

# Plot Polynomial R² Scores
plt.plot(degrees, poly_r2_scores, marker='o', linestyle='--', color='g', label="Polynomial Regression")
plt.axhline(y=linear_r2, color='r', linestyle='-', label="Linear Regression R²")
plt.xlabel("Polynomial Degree")
plt.ylabel("Test R² Score")
plt.title("Polynomial Regression Performance")
plt.legend()
plt.show()

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_model(rf_model, X_train, X_test, y_train, y_test, name="Random Forest")
