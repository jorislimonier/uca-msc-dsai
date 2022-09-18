# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Exercise 1

def strong_password(password):
    nb_missing = 0
    if len(password) < 6:
        nb_missing += 1
    if not any([character.isdigit() for character in password]):
        nb_missing += 1
    if not any([character.islower() for character in password]):
        nb_missing += 1
    if not any([character.isupper() for character in password]):
        nb_missing += 1
    if not any([character in "!@#$%^&*()-+" for character in password]):
        nb_missing += 1
    if nb_missing == 0:
        return True
    else:
        return nb_missing



# Exercise 2
import pandas as pd
from datetime import datetime

id_dict = {}

def add_user(name, birth_date, password):
    # Check if each word in name is capitalized
    if not all([n[0].isupper() for n in name.split(' ')]):
        print("Capitalize name")
        return
    
    # Check if password is strong
    while type(strong_password(password)) == type(1):
        password = input("invalid password, please try again\n")
    
    # Check if date is valid, else exit function
    try:
        pd.to_datetime(bd)
    except Exception as e:
        print("Please enter a valid date")
        return

    # Add user to id_dict
    global id_dict
    id_key = " ".join([name, birth_date])
    # If a user with same name and birth date already exists, append current time to key
    if id_key in id_dict.keys():
        id_key += " (" + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ")"
    id_dict[id_key] = password

name = "John Doa"
pw = "aaaa1+S"
bd = "12/02/1998"

add_user(name, bd, pw)

# Exercise 3
import cv2
import os

# !gdown --id 1l-Dhk04SuXAs_5v52j_2KrvexIRGYF2O
image_bib = cv2.imread(os.getcwd() + '/content/image_test2.jpg')
resized_image = cv2.resize(image_bib, (400,400), interpolation = cv2.INTER_AREA)

def crop(x1, x2, y1, y2):
    if any([x1 < 0, x2 > 400, y1 < 0, y2 > 400, x1 > x2, y1 > y2]):
        print("Invalid crop coordinates")
        return
    return resized_image[x1:x2, y1:y2, :]

def swap(nb_squares):
    if 400%nb_squares == 0 and nb_squares**(1/2) == 0:
        
        squares = resized_image


def modify_image(operation, display_method):
    if operation == "crop":
        new_image = crop()
    elif operation == "swap":
        new_image = swap()
    else:
        print("Invalid operation name")
        return
    
    if display_method == "save":
        plt.imsave("fig.jpg", new_image)
    elif display_method == "show":
        plt.imshow("fig.jpg", new_image)
    else:
        print("Invalid display method")


# %%
resized_image


# %%

import matplotlib.pyplot as plt
plt.imshow(resized_image)


