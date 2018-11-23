import csv
import matplotlib.pyplot as plt
import numpy as np


# declare & define a function for creating lists from csv file.
def create_lists():
    m = []
    n = []
    with open("team.csv", "r", newline='') as csvfile:
        row = csv.DictReader(csvfile)
        for i in row:
            m = np.append(m, i['Age']).astype(np.float)
            n = np.append(n, i['Experience']).astype(np.float)

    return m, n


# declare & define a function for coefficient calculation
def calculate(l1, l2):
    n = np.size(l1)

    # means
    m_x, m_y = np.mean(l1), np.mean(l2)

    # Cross Deviation
    SS_x = 0
    SS_y = 0
    for i in range(n):
        SS_x += (l1[i] - m_x) * (l2[i] - m_y)
        SS_y += (l1[i] - m_x) * (l1[i] - m_x)

    # Coefficients
    cf1 = SS_x / SS_y
    cf2 = m_y - cf1 * m_x

    return cf1, cf2


# declare & define a function for plotting
def plot(x, y, a, b):
    plt.scatter(x, y, color="m", marker="o")
    reg_line = a * x + b
    plt.plot(x, reg_line, color="g")

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

# main
def main():
    x, y = create_lists()
    a, b, = calculate(x, y)
    plot(x, y, a, b)

main()
