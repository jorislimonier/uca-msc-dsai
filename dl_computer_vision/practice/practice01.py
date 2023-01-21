import numpy as np


class LogReg:
  def __init__(self) -> None:
    ...

  @staticmethod
  def sigmoid(z: np.ndarray) -> float:
    return 1 / (1 + np.exp(-z))

  @staticmethod
  def cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
    return -np.dot(y, np.log(y_hat)) - np.dot((1 - y), np.log(1 - y_hat))

  @staticmethod
  def gradient(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray):
    return np.dot(x.T, y_hat - y)

  def logistic_regression(
    self, coef, x, y, lr: float, epsilon: float, max_iter: int = 1e5
  ):
    prev_loss = 0
    t = 0
    loss_tracker = []

    while t < max_iter:
      if (t + 1) % 50000 == 0:
        lr *= 0.1  # Learning rate decay

      z = np.dot(coef, x.T)
      y_hat = self.sigmoid(z)

      loss = self.cross_entropy(y=y, y_hat=y_hat)
      loss_tracker.append(loss)

      if t % max_iter // 20 == 0:
        print(loss)

      if abs(loss - prev_loss) <= epsilon:  # Early stop when the loss stabilizes
        print("BREAKING")
        break

      prev_loss = loss
      grad = self.gradient(x=x, y=y, y_hat=y_hat)

      t += 1
      delta = lr * grad
      coef -= delta

    self.loss_tracker = loss_tracker

    if t == max_iter - 1:
      print("COMPLETED ALL ITERATIONS")
    return coef
