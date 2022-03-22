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
