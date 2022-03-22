import os

import numpy as np
import cv2

DIRECTORY_SAVE_VIDEO = "video/"


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


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box):
    frameSize = (video.shape[1], video.shape[0])

    dir = main_directory + DIRECTORY_SAVE_VIDEO
    create_folder(dir)

    directory = dir + directory
    create_folder(directory)

    final_name = directory + name + "_" + type + ".mp4"

    out = cv2.VideoWriter(final_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, frameSize, True)

    for i in range(video.shape[3]):
        img = video[:, :, :, i]
        landmark_image = landmarks[:, :, i]
        bounding_box_image = bounding_box[:, :, i]

        img = add_landmarks(landmark_image, img, [0, 0, 255])
        img = add_bounding_box(bounding_box_image, img, [0, 255, 0])

        out.write(img)

    out.release()
