# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <a href="https://colab.research.google.com/github/bkkaggle/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # Install

# %%
get_ipython().system('git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix')


# %%
import os
os.chdir('pytorch-CycleGAN-and-pix2pix/')


# %%
get_ipython().system('pip install -r requirements.txt')

# %% [markdown]
# # Datasets
# 
# Download one of the official datasets with:
# 
# -   `bash ./datasets/download_pix2pix_dataset.sh [cityscapes, night2day, edges2handbags, edges2shoes, facades, maps]`
# 
# Or use your own dataset by creating the appropriate folders and adding in the images. Follow the instructions [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md#pix2pix-datasets).

# %%
get_ipython().system('bash ./datasets/download_pix2pix_dataset.sh facades')

# %% [markdown]
# # Pretrained models
# 
# Download one of the official pretrained models with:
# 
# -   `bash ./scripts/download_pix2pix_model.sh [edges2shoes, sat2map, map2sat, facades_label2photo, and day2night]`
# 
# Or add your own pretrained model to `./checkpoints/{NAME}_pretrained/latest_net_G.pt`

# %%
get_ipython().system('bash ./scripts/download_pix2pix_model.sh facades_label2photo')

# %% [markdown]
# # Training
# 
# -   `python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA`
# 
# Change the `--dataroot` and `--name` to your own dataset's path and model's name. Use `--gpu_ids 0,1,..` to train on multiple GPUs and `--batch_size` to change the batch size. Add `--direction BtoA` if you want to train a model to transfrom from class B to A.

# %%
get_ipython().system('python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA')

# %% [markdown]
# # Testing
# 
# -   `python test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name facades_pix2pix`
# 
# Change the `--dataroot`, `--name`, and `--direction` to be consistent with your trained model's configuration and how you want to transform images.
# 
# > from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix:
# > Note that we specified --direction BtoA as Facades dataset's A to B direction is photos to labels.
# 
# > If you would like to apply a pre-trained model to a collection of input images (rather than image pairs), please use --model test option. See ./scripts/test_single.sh for how to apply a model to Facade label maps (stored in the directory facades/testB).
# 
# > See a list of currently available models at ./scripts/download_pix2pix_model.sh

# %%
get_ipython().system('ls checkpoints/')


# %%
get_ipython().system('python test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name facades_label2photo_pretrained')

# %% [markdown]
# # Visualize

# %%
import matplotlib.pyplot as plt

img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_fake_B.png')
plt.imshow(img)


# %%
img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_real_A.png')
plt.imshow(img)


# %%
img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_real_B.png')
plt.imshow(img)


