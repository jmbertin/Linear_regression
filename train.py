import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gradient descent parameters
LEARN_RATE = 0.01
ITERATIONS = 1000

def compute_cost(X, y, theta):
    """
    Compute the cost (the difference between predicted values and the actual values).
    Args: X : array_like - The dataset of shape (m x n+1).
          y : array_like - A vector of shape (m, ) for the values at a given data point.
          theta : array_like - The linear regression parameters. A vector of shape (n+1, )
    Returns: J : float - The value of the cost function.
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)
    return J

def gradient_descent(X, y, theta):
    """
    Performs gradient descent to learn theta parameters.
    Args: X : array_like - The dataset of shape (m x n+1).
          y : array_like - A vector of shape (m, ) for the values at a given data point.
          theta : array_like - The linear regression parameters. A vector of shape (n+1, )
    Returns: theta : array_like - The learned linear regression parameters. A vector of shape (n+1, ).
             cost_history : list - List for the values of the cost function after each iteration.
    """
    m = len(y)
    cost_history = np.zeros(ITERATIONS)
    for i in range(ITERATIONS):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (LEARN_RATE / m) * X.transpose().dot(errors)
        theta = theta - sum_delta
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history


def plot_data(X, y, theta, mean, std):
    """
    Plot the data and the linear regression model.
    Args: X : array_like - The dataset of shape (m x n+1).
          y : array_like - A vector of shape (m, ) for the values at a given data point.
          theta : array_like - The linear regression parameters. A vector of shape (n+1, ).
          mean : float - The mean value of the data.
          std : float - The standard deviation of the data.
    Returns: None, but a plot is displayed.
    """
    # Draw the data
    plt.scatter(X, y)

    # Draw the regression line
    x_line = np.linspace(min(X), max(X), 1000)
    y_line = theta[0] + theta[1] * ((x_line - mean) / std)
    plt.plot(x_line, y_line, 'r')

    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Price vs Mileage')
    plt.show()

def main():
    """
    The main entry point of the program.
    This function handles the overall process of reading the dataset, preparing the data,
    training the model using gradient descent, saving the model parameters and plotting the results.
    """
    try:
        data = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("Unable to train model - dataset is not available")
        return

    X = np.array(data['km']).reshape(-1,1)
    y = np.array(data['price'])

    mean = np.mean(X)
    std = np.std(X)

    X_norm = (X - mean) / std
    X_norm = np.append(np.ones((X_norm.shape[0], 1)), X_norm, axis=1)

    theta = np.zeros(2)
    theta, _ = gradient_descent(X_norm, y, theta)

    np.savetxt('theta.csv', theta, delimiter=',')
    np.save('mean.npy', mean)
    np.save('std.npy', std)

    # Display plot (bonus)
    plot_data(X, y, theta, mean, std)

if __name__ == "__main__":
    main()
