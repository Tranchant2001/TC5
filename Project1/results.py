# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 23:05:52 2023

@author: jules
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn

t = np.array([15.702657469977156, 15.63635097144863,
              15.602166429392096, 15.581052501915442,
              15.574243686349192])

N = np.array([150, 200, 250, 300, 350])

h = 1 / N

plt.scatter(N, t)

# Create a function to perform linear regression for a given t0
def get_b(t0):
    # Calculate N based on the model N = -ln(t - t0)
    N_model = -np.log(t - t0)

    # Reshape data for regression
    N_model = N_model.reshape(-1, 1)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(N_model, N)

    # Get the coefficient "b"
    b = model.intercept_

    return b

alpha = 0.0001  # Initial step size
num_iter = 0
t0 = 15.5
b = get_b(t0)
b0 = b
cost = b**2

while num_iter < 100:
    print("Iteration:", num_iter, "t0:", t0, "b:", b)
    
    # Calculate a new step size based on the magnitude of b
    step_size = alpha   # Add a small constant to avoid division by zero
    
    t0 -= step_size * b
    num_iter += 1
    b = get_b(t0)
    cost = b**2
    
    if np.abs(b)< 0.01 :
        break

print("Estimated t0:", t0)
    
# Create a scatter plot of t against N
plt.scatter(N, t, label='Data')

# Add a horizontal line for the estimated t0
plt.axhline(y=t0, color='r', linestyle='--', label='Estimated t0')

# Set labels and legend
plt.xlabel('t')
plt.ylabel('N')
plt.legend()

# Show the plot
plt.show()