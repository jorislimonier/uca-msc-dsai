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
