import numpy as np
import pandas as pd
from scipy.optimize import minimize


class LogisticRegressionEM:
  """
  Logistic regression model using the EM algorithm.
  """

  def __init__(self, X: np.ndarray, y: np.ndarray, missing_mask: np.ndarray):
    """
    Initialize the class with feature matrix, labels, and missing mask.

    Parameters:
        X (np.ndarray): 2D array of shape (n, p) representing the feature matrix.
        y (np.ndarray): 1D array of shape (n,) representing the labels.
        missing_mask (np.ndarray): 2D array of shape (n, p) that is True where the corresponding value in X is missing, and False otherwise.
    """
    self.X = X
    self.y = y
    self.missing_mask = missing_mask
    self.n, self.p = X.shape
    self.beta = np.random.normal(size=self.p)

  def sigmoid(self, z: np.ndarray):
    """
    Compute the sigmoid function.

    Parameters:
        z (np.ndarray): 1D array of shape (n,).

    Returns:
        np.ndarray: 1D array of shape (n,) representing the sigmoid function applied to z.
    """
    return 1 / (1 + np.exp(-z))

  def complete_data_log_likelihood(self, beta: np.ndarray):
    """
    Compute the log-likelihood of the complete data (observed and estimated missing values) given the parameter estimates.

    Parameters:
        beta (np.ndarray): 1D array of shape (p,) representing the parameter estimates.

    Returns:
        float: The log-likelihood.
    """
    z = np.dot(self.X, beta)
    p = self.sigmoid(z)
    y_hat = np.round(p)
    log_likelihood = np.sum(y_hat == self.y)
    return log_likelihood

  def E_step(self, beta: np.ndarray):
    """
    Estimate the missing data based on the current parameter estimates.

    Parameters:
        beta (np.ndarray): 1D array of shape (p,) representing the current parameter estimates.
    """
    z = np.dot(self.X, beta)
    p = self.sigmoid(z)
    self.missing_data = np.random.binomial(1, p, size=self.missing_mask.shape)
    self.missing_data[~self.missing_mask] = self.y[~self.missing_mask]

  def M_step(self):
    """
    Update the parameter estimates based on the complete data (observed and estimated missing values) using maximum likelihood estimation.
    """
    self.beta = minimize(self.complete_data_log_likelihood, self.beta, method="BFGS").x

  def fit(self, max_iter=100):
    """
    Fit the logistic regression model using the EM algorithm.

    Parameters:
        max_iter (int): maximum number of iterations.
    """
    for _ in range(max_iter):
      self.E_step(self.beta)
      self.M_step()

  def predict(self, X):
    """
    Predict the labels for the given feature matrix.

    Parameters:
        X (np.ndarray): 2D array of shape (n, p) representing the feature matrix.

    Returns:
        np.ndarray: 1D array of shape (n,) representing the predicted labels.
    """
    z = np.dot(X, self.beta)
    p = self.sigmoid(z)
    return np.round(p)
