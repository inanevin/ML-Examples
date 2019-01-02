import numpy as num
import csv
import math
import matplotlib.pyplot as plt

with open("teams_comb.csv") as f:
    csv_list = list(csv.reader(f))

loocv_hat = num.array([])
age_list = num.array([])
exp_list = num.array([])
pow_list = num.array([])
salary_list = num.array([])
titles = []
titles_sorted = []

for row in csv_list:
    if row != csv_list[0]:
        age_list = num.append(age_list, int(row[4]))
        exp_list = num.append(exp_list, int(row[6]))
        pow_list = num.append(pow_list, float(row[7]))
        salary_list = num.append(salary_list, int(row[8]))
    else:
        titles.append(row[4])
        titles.append(row[6])
        titles.append(row[7])

ones = num.ones((1, len(age_list)))
mse_loocv = 0

for i in range(len(age_list)):  # Do the regression for each data point.

    # At each iteration, the i'th point becomes the test data and the rest becomes the train data.
    # We compute each prediction and store them.

    X = num.vstack((ones, age_list, exp_list, pow_list)).T
    Y = salary_list.T
    print(Y )

    X_test = X[i]   # The i'th point becomes the test data.
    Y_test = Y[i]
    X_train = num.delete(X, i, 0)   # The rest becomes the train data.
    Y_train = num.delete(Y, i, 0)

    # Compute the coefficients just like we did in LAB3.
    coefficients = num.linalg.inv(num.dot(X_train.T, X_train))
    coefficients = num.dot(coefficients, X_train.T)
    coefficients = num.dot(coefficients, Y_train)

    Y_hat = num.dot(X_test, coefficients)   # Compute the prediction using the coefficients
    loocv_hat = num.append(loocv_hat, Y_hat)  # Append the prediction into the array "loocv_hat" for future plotting
    mse_loocv = mse_loocv + math.pow((Y_hat-Y_test), 2)   # Sum the squared errors

mse_loocv = mse_loocv / len(age_list)   # Average the sum of squared errors

# Computing the regression one last time, as instructed in Task 2
coefficients = num.linalg.inv(num.dot(X.T, X))
coefficients = num.dot(coefficients, X.T)
coefficients = num.dot(coefficients, Y)

Y_hat = num.dot(X, coefficients)
mse_all = num.mean(num.square(Y_hat-Y))

print(mse_loocv)
print(mse_all)

# Plotting the results
plt.title("Residual Error Plot")
plt.scatter(Y_hat, Y_hat-Y)
plt.scatter(loocv_hat, loocv_hat-Y)
plt.hlines(y=0, xmin=0, xmax=40000, linewidth=2)
plt.show()

