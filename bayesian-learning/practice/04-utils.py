import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm, uniform, binom
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal


def laplace_solution(params, other_params, data, lik, priors):
  def evaluate_log_post(params, other_params, data, lik, priors):
    model_list = {"gaussian": norm, "uniform": uniform, "binomial": binom}

    model_lik = model_list[lik]

    # Computing log-priors
    log_prior = 0
    for i, mod in enumerate(priors):
      log_prior += model_list[mod[0]].logpdf(params[i], *mod[1])

    # Computing log-likelihood
    if lik == "gaussian":
      # Dirty trick for guaranteeing positive variance
      params[-1] = np.abs(params[-1])

    if len(other_params) > 0:
      log_lik = np.sum(
        [model_list[lik].logpdf(point, *(params, other_params)) for point in data]
      )
    else:
      log_lik = np.sum([model_list[lik].logpdf(point, *params) for point in data])
    return -(log_lik + log_prior)

  minimum = minimize(
    evaluate_log_post, params, args=(other_params, data, lik, priors), method="BFGS"
  )
  print(minimum)
  return [minimum.x, minimum.hess_inv]


def laplace_solution_regression(expression, data, lik, priors):
  model_list = {"gaussian": norm, "uniform": uniform, "binomial": binom}

  def evaluate_log_post(params, var_names, data, lik, priors):
    model_list = {"gaussian": norm, "uniform": uniform, "binomial": binom}
    model_lik = model_list[lik]

    # Computing log-priors
    log_prior = 0
    for i, mod in enumerate(priors):
      log_prior += model_list[mod[0]].logpdf(params[i], *mod[1])

    # Evaluating expression
    target, predictors = var_names[0], var_names[1]

    mu = np.ones(len(data[predictors[0]])) * params[0]

    for i in range(len(predictors)):
      mu += params[i + 1] * data[predictors[i]].values

    sigma = np.abs(params[-1])

    t = data[target].values
    N = len(t)

    log_lik = np.sum(
      [model_list["gaussian"].logpdf(t[i], mu[i], sigma) for i in range(N)]
    )
    return -(log_lik + log_prior)

  collapsed_expression = expression.replace(" ", "")
  target, independent = collapsed_expression.split("~")
  independent = independent.split("+")
  var_names = [target, independent]

  params = []
  for i in range(len(priors)):
    params.append(model_list[priors[i][0]].rvs(*priors[i][1]))

  minimum = minimize(
    evaluate_log_post, params, args=(var_names, data, lik, priors), method="BFGS"
  )
  print(minimum)
  return [minimum.x, minimum.hess_inv]


def post_sample_Laplace(solution, N_sample):
  posterior_samples = multivariate_normal.rvs(solution[0], solution[1], size=N_sample)
  return posterior_samples


def posterior_stats(solution, names, plot=False):
  posterior_samples = post_sample_Laplace(solution, 1000)
  post_quantiles = np.quantile(posterior_samples, q=[0.075, 0.925], axis=0)
  # sd
  post_sd = np.std(posterior_samples, axis=0)
  # mean
  post_mean = np.mean(posterior_samples, axis=0)
  summary_stats = [post_mean, post_sd, post_quantiles[0, :], post_quantiles[1, :]]
  summary_stats = pd.DataFrame(summary_stats).transpose()
  summary_stats.columns = ["mean", "SD", "7.5%", "92.5%"]
  summary_stats.index = names
  # summary_stats.rename(index=list(names), inplace=True)
  print(summary_stats)
  if plot:
    boxes = []
    for i in range(len(post_mean)):
      boxes.append(posterior_samples[:, i])
    plt.boxplot(boxes, vert=0)
    plt.yticks(range(len(post_mean) + 1)[1:], names)
    plt.axvline(x=0, color="black", alpha=0.1)
    plt.show()


def posterior_plot_univariate_regression(
  solution, x_range, data, center=0, names=["x", "y"], N_samples=500
):
  post_samples = post_sample_Laplace(solution, N_samples)
  degree = post_samples.shape[1] - 1

  post_mean = np.mean(post_samples, axis=0)

  mean_prediction = np.zeros(len(x_range))
  for i in range(degree):
    mean_prediction += post_mean[i] * x_range**i

  plt.plot(x_range + center, mean_prediction, lw=2, color="black")
  plt.title("regression result")
  plt.ylabel(names[1])
  plt.xlabel(names[0])

  list_model_samples = []
  # sampling from the posterior to get a predictive interval
  for n in range(N_samples):
    mod_sample = np.zeros(len(x_range))
    for i in range(degree):
      mod_sample += post_samples[n, i] * x_range**i
    list_model_samples.append(mod_sample)

  prediction_noise = []
  for i, mod_sample in enumerate(list_model_samples):
    prediction_noise.append(norm.rvs(mod_sample, post_samples[i, -1]))

  # Plotting the uncertainty
  for i in range(N_samples):
    plt.scatter(x_range + center, prediction_noise[i], alpha=0.05, color="grey")
    plt.scatter(x_range + center, list_model_samples[i], alpha=0.05, color="green")

  plt.scatter(data[names[0]] + center, data[names[1]])
