from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import numpy as np


class LSM:
  def __init__(self):
    self.track_negloklik = []

  def negloglik(self, params, X):
    """
    Compute the negative log-likelihood of the latent space model.

    Parameters
    ----------
    params : array-like
      The last parameter is the alpha parameter, the rest are the
      2D coordinates of the latent positions.
    X : array-like
      The adjacency matrix of the graph.

    Returns
    -------
    ll : float
      The negative log-likelihood of the latent space model.
    """
    *latent_pos, alpha = params

    # Reshape the latent positions
    latent_pos = np.array(latent_pos).reshape(-1, 2)
    # print("-" * 10)

    # Compute the distance between each pair of points
    latent_dist = distance_matrix(latent_pos, latent_pos)

    # Compute the negative log-likelihood
    ll_mat = X * (alpha - latent_dist) - np.log(1 + np.exp(alpha - latent_dist))
    
    # print(ll_mat)
    nll = -np.sum(ll_mat)
    self.track_negloklik.append(nll)

    return nll

  def compute_lsm(self, X):
    """
    Compute the latent space model for a given matrix X.
    """
    latent_pos_size = (len(X), 2)
    latent_pos_init = np.random.normal(size=latent_pos_size)
    alpha_init = 0.5
    params = np.concatenate([latent_pos_init.flatten(), [alpha_init]])
    out = minimize(fun=self.negloglik, x0=params, args=(X,))

    return out
