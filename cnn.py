import cv2
from mtcnn.mtcnn import MTCNN

import save_video as sv
import viola_jones as vj

DIRECTORY_CNN = "cnn/"


def find_faces(img):

    detector = MTCNN()
    faces = detector.detect_faces(img)

    coordinates = []
    for face in faces:
        column, row, width, height = face["box"]
        coordinate = vj.get_four_vertices(column, row, width, height)
        coordinates.append(coordinate)
        img = sv.add_bounding_box(coordinate, img, [48, 88, 247])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, coordinates


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box):
    frameSize = (video.shape[1], video.shape[0])

    dir = main_directory + DIRECTORY_CNN
    sv.create_folder(dir)

    directory = dir + directory
    sv.create_folder(directory)

    final_name = directory + name + "_" + type + ".mp4"

    out = cv2.VideoWriter(final_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, frameSize, True)

    for i in range(video.shape[3]):
        img = video[:, :, :, i]

        img, coordinates_found_bounding_box = find_faces(img)

        landmark_image = landmarks[:, :, i]
        bounding_box_image = bounding_box[:, :, i]

        img = sv.add_landmarks(landmark_image, img, [0, 0, 255])
        img = sv.add_bounding_box(bounding_box_image, img, [0, 255, 0])

        out.write(img)

    out.release()
