# Linear_regression

Linear_regression is a project undertaken as part of the 42 school curriculum. This repository contains Python scripts to train a linear regression model on a dataset of car prices and to use the trained model with a gradient descent algorithm to estimate car prices based on mileage. The project also includes scripts for plotting the training data and regression line and evaluating the model using common metrics.
The aim of this project is to introduce to the basic concept behind machine learning.

----
## Files Description

1. train.py: This script handles the training process of the linear regression model using gradient descent. It reads a dataset from a CSV file, normalizes the data, runs gradient descent to find the optimal parameters, and saves the model parameters and normalization values.

2. training.py: This script estimates the price of a car given its mileage using the trained model parameters. It loads the model parameters and normalization values, takes mileage as input, and outputs the estimated price.

3. evaluate.py: This script evaluates the performance of the trained model using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared. It computes these metrics using the actual prices and the prices predicted by the model on the training dataset.

----
## Requirements

- Python 3.x
- NumPy
- pandas
- Matplotlib

Install the necessary Python packages by running the following command:
``pip install numpy pandas matplotlib``

----

## Usage

1. Training the model:

- Place your dataset in a CSV file named data.csv with columns "km" for mileage and "price" for car prices.
- Run the training script:
``python train.py``
- The trained model parameters will be saved to theta.csv, and the normalization values will be saved to mean.npy and std.npy.
- A plot of the training data and the regression line will be displayed.

2. Estimating car prices:

- Run the estimation script:
``python training.py``
- Enter the mileage of the car when prompted to get the estimated price.

3. Evaluating the model:

- Run the evaluation script:
``python evaluate.py``

- The script will print the MAE, RMSE, and R-squared values computed on the training dataset.

----

## Contribution
If you encounter any bugs or wish to add features, please feel free to open an issue or submit a pull request.

