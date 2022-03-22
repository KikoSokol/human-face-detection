import os

import numpy as np


def add_bounding_box(bounding_box, image, color):
    for up in range(int(bounding_box[0][1]), int(bounding_box[1][1]) + 1):
        image[up][int(bounding_box[0][0])] = np.array(color)

    for down in range(int(bounding_box[2][1]), int(bounding_box[3][1]) + 1):
        image[down][int(bounding_box[2][0])] = np.array(color)

    for left in range(int(bounding_box[0][0]), int(bounding_box[2][0]) + 1):
        image[int(bounding_box[0][1])][left] = np.array(color)

    for right in range(int(bounding_box[1][0]), int(bounding_box[3][0]) + 1):
        image[int(bounding_box[1][1])][right] = np.array(color)

    return image


def add_landmarks(landmarks, image, color):
    for i in landmarks:
        image[int(i[1])][int(i[0])] = np.array(color)

    return image


def create_folder(name):
    if os.path.isdir(name) is True:
        return None
    else:
        os.mkdir(name[0: len(name) - 1])


def create_directory_and_get_file_name(main_directory, category_directory, directory, name, type):
    dir = main_directory + category_directory
    create_folder(dir)

    directory = dir + directory
    create_folder(directory)

    return directory + name + "_" + type + ".mp4"
