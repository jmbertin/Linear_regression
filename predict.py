import numpy as np

def estimate_price(mileage, theta):
    """
    Estimate the price of the car using the model parameters (theta) and the mileage.
    Args: mileage : float - The mileage of the car.
          theta : array_like - The parameters of the model. A vector of shape (n+1, ).
    Returns: price : float - The estimated price of the car.
    """
    return theta[0] + (theta[1] * mileage)

def main():
    """
    The main entry point of the program.
    This function handles the overall process of loading the model parameters, taking the mileage input,
    normalizing the mileage, estimating the price and printing the estimated price.
    """
    try:
        theta = np.loadtxt('theta.csv', delimiter=',')
    except OSError:
        theta = np.zeros(2)
    try:
        mean = np.load('mean.npy')
    except FileNotFoundError:
        mean = 0.0
    try:
        std = np.load('std.npy')
    except FileNotFoundError:
        std = 1.0
    try:
        mileage = float(input("Enter a mileage: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    mileage_norm = (mileage - mean) / std
    price = estimate_price(mileage_norm, theta)
    if price < 0:
        price = 0
    print(f"The estimated price for a vehicle with {mileage} kilometers is {price:.2f}")

if __name__ == "__main__":
    main()
