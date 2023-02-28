<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

**Table of Contents** _generated with [DocToc](https://github.com/thlorenz/doctoc)_

- [Federated Learning & Data Privacy](#federated-learning--data-privacy)
  - [Evaluation](#evaluation)
  - [Labs](#labs)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Federated Learning & Data Privacy

## Evaluation

- 3 marks
  - 50% Final exam: paper and pen open questions
  - 25% Minitest 10 or 15 min
  - 25% Participation to the labs

## Labs

1. Intro to federated learning & framework implementation
1.
1. Personalization
1. Security, attacks

## Minitests

### Minitest 1

#### Define FL

FL is a way to train a machine learning model while keeping the datasets local at the users (clients over the internet)

#### When should we use FL?

- Cost of data collection
- Data privacy (GDPR)

### Minitest 2

#### What is per-client fairness (PCF) and per-sample fairness (PSF)?

PCF: each client should have the same importance
PSF: each sample should have the same importance

#### How is it implemented in FL?
Let $K$ be the number of clients and $n_k$ the number of samples of client $k$. Let $\{\xi_{k,l}\}_{l=1}^{n_k}$ be the samples of client $k$.

Client objective function:
$$
F_k(w) = \frac{1}{n_k} \sum_{l=1}^{n_k} \ell(w, \xi_{k,l})
$$
Global objective function:
$$
F(w) = \sum_{k=1}^K p_k F_k(w)
$$

### Minitest 3
#### Question 1

$$
F(w) = \sum_{i=1}^M \frac{1}{M} F_i(w)
$$
so $\alpha_i = \frac{1}{M}$.

For $k = 1, \ldots M$: \
$\quad$ $A_k$ is the set of available clients at round $k$ \
$\quad$ Send weights to the server\
$\quad$ For each client $i \in A_k$:\
$\quad$ $\quad$ Perform $E$ local steps.\
$\quad$ $\Delta_k = \sum_{i \in A_k} q_i \Delta_k^{(i)}$, with $q_i = \frac{1}{M \pi_i}$\
$\quad$ $w_{k+1, 0} = w_{k, 0} + \eta_s \Delta_k$

#### Question 2
Yes, when number of iterations if limited and if converging to $F_B$ may be faster than converging to $F$.