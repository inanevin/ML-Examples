import numpy as num
import csv
import sklearn.linear_model as ln
import matplotlib.pyplot as plt

# Multiple Linear Regression for LAB3, written by Çınar Gedizlioğlu

# Opening and reading from the .csv file.
with open("team.csv") as f:
    csv_list = list(csv.reader(f))

age_list = num.array([])
exp_list = num.array([])
pow_list = num.array([])
salary_list = num.array([])
titles = []
titles_sorted = []

# Extract each column of the .csv file to a different numpy array except the first row
# Also extract the first row to the array called "titles"
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

# I used numpy's "vstack" function to combine all of the independent variables into one matrix.
# I need all the vectors as column vectors, so I take the transpose of both X and Y.
X = num.vstack((ones, age_list, exp_list, pow_list)).T
Y = salary_list.T



# These three lines are the calculations for the coefficients given in the instructions.
coefficients = num.linalg.inv(num.dot(X.T, X))
coefficients = num.dot(coefficients, X.T)
coefficients = num.dot(coefficients, Y)

# Calculating the predictions is now a simple dot product.
Y_hat = num.dot(X, coefficients)

# Alternatively, instead of the last 5 lines of code, you can use the sklearn package:
#
# reg = ln.LinearRegression()
# reg.fit(X, Y)
# reg.predict(X)
#
# The coefficients can then be found in reg.coef_

# This loop is for the last task of LAB3, which is to sort the TITLES of the variables in order of importance.
temp_coef = coefficients
for i in range(len(coefficients)-1):
    titles_sorted.append(titles[temp_coef.argmax()-1])
    titles.remove(titles[temp_coef.argmax() - 1])
    temp_coef = num.delete(temp_coef, temp_coef.argmax())

print(titles_sorted)

plt.title("Residual Error Plot")
plt.scatter(Y_hat, Y_hat-Y)

plt.hlines(y=0, xmin=0, xmax=20000, linewidth=2)
plt.show()

# Remember, this code does NOT use different data sets for training and testing the data.
# You will need small alterations to this code if you want to implement such a multiple linear regression method.
