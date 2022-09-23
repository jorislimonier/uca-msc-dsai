<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [EMNIST Dataset](#emnist-dataset)
  - [Introduction](#introduction)
  - [Instructions](#instructions)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# EMNIST Dataset

## Introduction

Split MNIST dataset among `n_clients` as follows:
1) sort the data by label
2) divide it into `n_clients * n_classes_per_client` shards, of equal size
3) assign each of the `n_clients` with `n_classes_per_client` shards

Inspired by the split in [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

## Instructions

Run `generate_data.py` with a choice of the following arguments:

- ```--n_clients```: number of tasks/clients, written as integer
- ```--n_iid```: if selected the dataset is slit in a non i.i.d. fashion, otherwise it is i.i.d
- ```--seed``` : seed to be used before random sampling of data; default=``12345``
