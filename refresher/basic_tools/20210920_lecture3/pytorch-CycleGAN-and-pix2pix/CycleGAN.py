# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <a href="https://colab.research.google.com/github/bkkaggle/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# Take a look at the [repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for more information
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
# -   `bash ./datasets/download_cyclegan_dataset.sh [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos]`
# 
# Or use your own dataset by creating the appropriate folders and adding in the images.
# 
# -   Create a dataset folder under `/dataset` for your dataset.
# -   Create subfolders `testA`, `testB`, `trainA`, and `trainB` under your dataset's folder. Place any images you want to transform from a to b (cat2dog) in the `testA` folder, images you want to transform from b to a (dog2cat) in the `testB` folder, and do the same for the `trainA` and `trainB` folders.

# %%
get_ipython().system('bash ./datasets/download_cyclegan_dataset.sh horse2zebra')

# %% [markdown]
# # Pretrained models
# 
# Download one of the official pretrained models with:
# 
# -   `bash ./scripts/download_cyclegan_model.sh [apple2orange, orange2apple, summer2winter_yosemite, winter2summer_yosemite, horse2zebra, zebra2horse, monet2photo, style_monet, style_cezanne, style_ukiyoe, style_vangogh, sat2map, map2sat, cityscapes_photo2label, cityscapes_label2photo, facades_photo2label, facades_label2photo, iphone2dslr_flower]`
# 
# Or add your own pretrained model to `./checkpoints/{NAME}_pretrained/latest_net_G.pt`

# %%
get_ipython().system('bash ./scripts/download_cyclegan_model.sh horse2zebra')

# %% [markdown]
# # Training
# 
# -   `python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan`
# 
# Change the `--dataroot` and `--name` to your own dataset's path and model's name. Use `--gpu_ids 0,1,..` to train on multiple GPUs and `--batch_size` to change the batch size. I've found that a batch size of 16 fits onto 4 V100s and can finish training an epoch in ~90s.
# 
# Once your model has trained, copy over the last checkpoint to a format that the testing model can automatically detect:
# 
# Use `cp ./checkpoints/horse2zebra/latest_net_G_A.pth ./checkpoints/horse2zebra/latest_net_G.pth` if you want to transform images from class A to class B and `cp ./checkpoints/horse2zebra/latest_net_G_B.pth ./checkpoints/horse2zebra/latest_net_G.pth` if you want to transform images from class B to class A.
# 

# %%
get_ipython().system('python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan')

# %% [markdown]
# # Testing
# 
# -   `python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout`
# 
# Change the `--dataroot` and `--name` to be consistent with your trained model's configuration.
# 
# > from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix:
# > The option --model test is used for generating results of CycleGAN only for one side. This option will automatically set --dataset_mode single, which only loads the images from one set. On the contrary, using --model cycle_gan requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at ./results/. Use --results_dir {directory_path_to_save_result} to specify the results directory.
# 
# > For your own experiments, you might want to specify --netG, --norm, --no_dropout to match the generator architecture of the trained model.

# %%
get_ipython().system('python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout')

# %% [markdown]
# # Visualize

# %%
import matplotlib.pyplot as plt

img = plt.imread('./results/horse2zebra_pretrained/test_latest/images/n02381460_1010_fake.png')
plt.imshow(img)


# %%
import matplotlib.pyplot as plt

img = plt.imread('./results/horse2zebra_pretrained/test_latest/images/n02381460_1010_real.png')
plt.imshow(img)


