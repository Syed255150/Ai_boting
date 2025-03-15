import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate Sample Data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Linear Regression Model
def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    # Plot the regression line
    plt.scatter(X, y, color='blue', label="Data")
    plt.plot(X, y_pred, color='red', label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression")
    plt.show()
    
    # Print Model Parameters
    intercept = model.intercept_[0]
    coefficient = model.coef_[0][0]
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Linear Regression - Intercept: {intercept}")
    print(f"Linear Regression - Coefficient: {coefficient}")
    print(f"Linear Regression - Mean Squared Error: {mse}")
    print(f"Linear Regression - R² Score: {r2}")
    
    return r2

linear_r2 = linear_regression(X, y)

# Polynomial Regression Model
def polynomial_regression(X, y, degree=2):
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X, y)
    
    y_pred = poly_model.predict(X)
    
    # Sort data for smooth plotting
    X_sorted, y_pred_sorted = zip(*sorted(zip(X.flatten(), y_pred.flatten())))
    
    # Plot the regression curve
    plt.scatter(X, y, color='blue', label="Data")
    plt.plot(X_sorted, y_pred_sorted, color='red', label=f"Polynomial Degree {degree}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Polynomial Regression (Degree {degree})")
    plt.show()
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Polynomial Degree {degree} - Mean Squared Error: {mse}")
    print(f"Polynomial Degree {degree} - R² Score: {r2}")
    
    return r2

# Run polynomial regression with multiple degrees (2 to 5)
degrees = list(range(2, 6))
r2_scores = [polynomial_regression(X, y, degree=d) for d in degrees]

# Plot R² Score vs. Degree
plt.plot(degrees, r2_scores, marker='o', linestyle='--', color='g', label="Polynomial Regression")
plt.axhline(y=linear_r2, color='r', linestyle='-', label="Linear Regression R²")
plt.xlabel("Polynomial Degree")
plt.ylabel("R² Score")
plt.title("Model Performance for Different Polynomial Degrees")
plt.legend()
plt.show()

# Save results to a file
with open("regression_results.txt", "w") as f:
    f.write(f"Linear Regression - R² Score: {linear_r2}\n")
    for d, r2 in zip(degrees, r2_scores):
        f.write(f"Polynomial Degree {d} - R² Score: {r2}\n")

print("Results saved to 'regression_results.txt'.")
