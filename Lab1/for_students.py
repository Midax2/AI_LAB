import numpy as np
import matplotlib.pyplot as plt
import random
from data import get_data, inspect_data, split_data


def calculate_MSE(m, actual_output, weights, theta) -> float:
    y_new_train = [0.0] * m
    for i, weight in enumerate(weights):
        var = theta[1] * weight + theta[0]
        var2 = (var - actual_output[i]) ** 2
        y_new_train[i] = var2
    return (1 / m) * np.sum(y_new_train)


def delta_MSE(y, x, theta):
    m = len(y)
    X = np.zeros((m, 2))
    X[:, 0] = 1
    X[:, 1:] = x.reshape(-1, 1)
    y_help = y.reshape(-1, 1)
    X_transposed = np.transpose(X)
    step = (2 / m) * X_transposed
    step2 = np.subtract(np.matmul(X, theta), y_help)
    return np.matmul(step, step2)


def batch_gradient_descent(y, x):
    learning_rate = 0.01
    new_theta = [random.random(), random.random()]
    theta = np.array(new_theta).reshape(-1, 1)
    prev_MSE = 0.0
    while True:
        theta = np.subtract(theta, learning_rate * delta_MSE(y, x, theta))
        t1 = theta.item(1)
        t0 = theta.item(0)
        curr_MSE = calculate_MSE(len(y), y, x, theta)
        if curr_MSE == prev_MSE:
            break
        prev_MSE = curr_MSE

    return float(t0), float(t1)


def closed_form_solution(y, x):
    m = len(y)
    n = 2
    X = np.zeros((m, n))
    X[:, 0] = 1
    X[:, 1:] = x.reshape(-1, 1)
    y_help = y.reshape(-1, 1)
    X_transposed = np.transpose(X)
    step = np.matmul(X_transposed, X)
    reverse_help = np.linalg.inv(step)
    step2 = np.matmul(reverse_help, X_transposed)
    theta = np.matmul(step2, y_help)
    t1 = theta.item(1)
    t0 = theta.item(0)
    return float(t0), float(t1)


def normalization(array):
    temp = [0.0] * len(array)
    arr_avg = np.mean(array)
    arr_std = np.std(array)
    for i, item in enumerate(array):
        temp[i] = (array[i] - arr_avg) / arr_std
    return np.array(temp)


def main():
    data = get_data()
    inspect_data(data)

    train_data, test_data = split_data(data)

    # Simple Linear Regression
    # predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
    # y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

    # We can calculate the error using MSE metric:
    # MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

    # get the columns
    y_train = train_data['MPG'].to_numpy()
    x_train = train_data['Weight'].to_numpy()

    y_test = test_data['MPG'].to_numpy()
    x_test = test_data['Weight'].to_numpy()

    # TODO: calculate closed-form solution
    theta_best = closed_form_solution(y_train, x_train)

    # TODO: calculate error
    print("Closed-form solution:")
    print("MSE dla y_train wynosi " + str(calculate_MSE(len(y_train), y_train, x_train, theta_best)))
    print("MSE dla y_test wynosi " + str(calculate_MSE(len(y_test), y_test, x_test, theta_best)))
    # plot the regression line
    x = np.linspace(min(x_test), max(x_test), 100)
    y = float(theta_best[0]) + float(theta_best[1]) * x
    plt.plot(x, y)
    plt.scatter(x_test, y_test)
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.show()

    # TODO: standardization
    y_train_norm = normalization(y_train)
    x_train_norm = normalization(x_train)
    # y_test_norm = normalization(y_test)
    # x_test_norm = normalization(x_test)
    print(x_train_norm)
    # TODO: calculate theta using Batch Gradient Descent
    theta_best = batch_gradient_descent(y_train_norm, x_train_norm)
    temp1 = theta_best[1] * np.std(y_train) / np.std(x_train)
    temp0 = np.mean(y_train) - temp1 * np.mean(x_train)
    theta_best = (temp0, temp1)

    # TODO: calculate error
    print("\nBatch Gradient Descent:")
    print("MSE dla y_train wynosi " + str(calculate_MSE(len(y_train), y_train, x_train, theta_best)))
    print("MSE dla y_test wynosi " + str(calculate_MSE(len(y_test), y_test, x_test, theta_best)))

    # plot the regression line
    x = np.linspace(min(x_test), max(x_test), 100)
    y = float(theta_best[0]) + float(theta_best[1]) * x
    plt.plot(x, y)
    plt.scatter(x_test, y_test)
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.show()


if __name__ == "__main__":
    main()
