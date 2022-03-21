import os

import numpy as np
import cv2


def add_bounding_box(bounding_box, image):
    for up in range(int(bounding_box[0][1]), int(bounding_box[1][1]) + 1):
        image[up][int(bounding_box[0][0])] = np.array([0, 255, 0])

    for down in range(int(bounding_box[2][1]), int(bounding_box[3][1]) + 1):
        image[down][int(bounding_box[2][0])] = np.array([0, 255, 0])

    for left in range(int(bounding_box[0][0]), int(bounding_box[2][0]) + 1):
        image[int(bounding_box[0][1])][left] = np.array([0, 255, 0])

    for right in range(int(bounding_box[1][0]), int(bounding_box[3][0]) + 1):
        image[int(bounding_box[1][1])][right] = np.array([0, 255, 0])

    return image


def add_landmarks(landmarks, image):
    for i in landmarks:
        image[int(i[1])][int(i[0])] = np.array([0, 0, 255])

    return image


def create_folder(name):
    if os.path.isdir(name) is True:
        return None
    else:
        os.mkdir(name[0: len(name) - 1])


def to_mp4(directory, name, type, video, landmarks, bounding_box):
    frameSize = (video.shape[1], video.shape[0])

    create_folder(directory)

    final_name = directory + name + "_" + type + ".mp4"

    out = cv2.VideoWriter(final_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, frameSize, True)

    for i in range(video.shape[3]):
        img = video[:, :, :, i]
        landmark_image = landmarks[:, :, i]
        bounding_box_image = bounding_box[:, :, i]

        img = add_landmarks(landmark_image, img)
        img = add_bounding_box(bounding_box_image, img)

        out.write(img)

    out.release()


FOLDER_WITH_NPZ = "viz_vzorka/"
FOLDER_WITH_MP4 = "mp4/"

npz_files = ["Kieran_Culkin_0.npz", "Liu_Ye_2.npz", "Maggie_Smith_3.npz", "Margaret_Thatcher_5.npz",
             "Marisa_Tomei_1.npz", "Martin_Sheen_3.npz", "Martin_Sheen_5.npz", "Matt_Anderson_2.npz",
             "Natalie_Stewart_2.npz", "Oscar_Elias_Biscet_0.npz"]

for file_name in npz_files:
    video_file = np.load(FOLDER_WITH_NPZ + file_name)
    file_name_without_suffix = file_name.split(".")[0]
    directory_name = FOLDER_WITH_MP4 + file_name_without_suffix + "/"
    to_mp4(directory_name, file_name_without_suffix, "ORIGINAL", video_file["colorImages_original"],
           video_file["landmarks2D"],
           video_file["boundingBox"])
    to_mp4(directory_name, file_name_without_suffix, "MEDIUM", video_file["colorImages_medium"],
           video_file["landmarks2D"],
           video_file["boundingBox"])
    to_mp4(directory_name, file_name_without_suffix, "SEVERE", video_file["colorImages_severe"],
           video_file["landmarks2D"],
           video_file["boundingBox"])
