# Exercise 1

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from datetime import datetime
import pandas as pd
from matplotlib.image import imsave


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

# !gdown --id 1l-Dhk04SuXAs_5v52j_2KrvexIRGYF2O
image_bib = cv2.imread(os.getcwd() + '/content/image_test2.jpg')
resized_image = cv2.resize(image_bib, (400, 400), interpolation=cv2.INTER_AREA)
plt.imsave("resized_img.jpg", resized_image)

def crop(x1, x2, y1, y2):
    if any([x1 < 0, x2 > 400, y1 < 0, y2 > 400, x1 > x2, y1 > y2]):
        print("Invalid crop coordinates")
        return
    return resized_image[x1:x2, y1:y2, :]


def divisers(nb_squares):
    nb_rows = 2
    while nb_rows <= nb_squares**(1/2):
        if nb_squares % nb_rows == 0:
            nb_col = int(nb_squares/nb_rows)
            return nb_rows, nb_col
        nb_rows += 1


def swap(nb_squares):
    if 400 % nb_squares != 0 or divisers(nb_squares) is None:
        print("Bad number of squares")
        return
    nb_rows, nb_col = divisers(nb_squares)
    global swapped_img
    swapped_img = np.zeros((400,400,3),dtype=np.uint8)
    for i in range(nb_rows):
        for j in range(nb_col):
            beginning_i = int(nb_col*i * 400/nb_squares)
            end_i = int(nb_col*(i+1) * 400/nb_squares)
            beginning_j = int(nb_rows*j * 400/nb_squares)
            end_j = int(nb_rows*(j+1) * 400/nb_squares)
            print(f"{beginning_i, end_i, beginning_j, end_j}")
            swapped_img[beginning_i:end_i, beginning_j:end_j, :] = resized_image[400-end_i: 400-beginning_i, 400-end_j: 400-beginning_j, :]
                
    plt.imsave(f"fig_{nb_squares}squares.jpg", swapped_img)
swap(400)


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
        return
