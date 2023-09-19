import numpy as np
import pandas as pd

def compute_metrics(y, y_pred):
    """
    Compute Mean Absolute Error and Root Mean Squared Error.
    Args: y : array_like - A vector of actual values.
          y_pred : array_like - A vector of predicted/estimated values.
    Returns: mae : float - Mean Absolute Error.
             rmse : float - Root Mean Squared Error.
    """
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    return mae, rmse

def compute_r2(y, y_pred):
    """
    Compute R-squared (coefficient of determination).
    Args: y : array_like - A vector of actual values.
          y_pred : array_like - A vector of predicted/estimated values.
    Returns: r2 : float - R-squared metric.
    """
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def main():
    """
    The main entry point of the program.
    This function handles the overall process of loading the model parameters,
    performing the predictions, computing the metrics and printing the metrics.
    """
    try:
        theta = np.loadtxt('theta.csv', delimiter=',')
        mean = np.load('mean.npy')
        std = np.load('std.npy')
        data = pd.read_csv('data.csv')
    except (OSError, FileNotFoundError):
        print("Impossible d'évaluer le modèle - les valeurs nécessaires ne sont pas disponibles ou invalides")
        return

    X = np.array(data['km']).reshape(-1,1)
    y = np.array(data['price'])

    X_norm = (X - mean) / std
    X_norm = np.append(np.ones((X_norm.shape[0], 1)), X_norm, axis=1)
    y_pred = X_norm.dot(theta)

    mae, rmse = compute_metrics(y, y_pred)
    r2 = compute_r2(y, y_pred)

    print("Mean Absolute Error (MAE): ", mae)
    print("Root Mean Squared Error (RMSE): ", rmse)
    print("R-squared: ", r2)

if __name__ == "__main__":
    main()
