import cv2
from mtcnn.mtcnn import MTCNN

import helper as hp

DIRECTORY_CNN = "cnn/"


def find_faces(img):

    detector = MTCNN()
    faces = detector.detect_faces(img)

    coordinates = []
    for face in faces:
        column, row, width, height = face["box"]
        coordinate = hp.get_four_vertices(column, row, width, height)
        coordinates.append(coordinate)
        img = hp.add_bounding_box(coordinate, img, [48, 88, 247])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, coordinates


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box):
    final_name = hp.create_directory_and_get_file_name(main_directory, DIRECTORY_CNN, directory, name, type)
    frameSize = (video.shape[1], video.shape[0])

    out = cv2.VideoWriter(final_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, frameSize, True)

    for i in range(video.shape[3]):
        img = video[:, :, :, i]

        img, coordinates_found_bounding_box = find_faces(img)

        landmark_image = landmarks[:, :, i]
        bounding_box_image = bounding_box[:, :, i]

        img = hp.add_landmarks(landmark_image, img, [0, 0, 255])
        img = hp.add_bounding_box(bounding_box_image, img, [0, 255, 0])

        out.write(img)

    out.release()
