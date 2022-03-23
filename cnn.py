import cv2
from mtcnn.mtcnn import MTCNN

import helper as hp

DIRECTORY_CNN = "cnn/"


def find_faces(img, correct_bounding_box):

    detector = MTCNN()
    faces = detector.detect_faces(img)

    tp = 0
    fp = 0
    fn = 0

    for face in faces:
        column, row, width, height = face["box"]
        coordinate = hp.get_four_vertices(column, row, width, height)

        iou = hp.compute_squares_iou(coordinate, correct_bounding_box)

        if iou > 0.5:
            tp += 1
            img = hp.add_bounding_box(coordinate, img, [48, 88, 247])
        else:
            fp += 1
            img = hp.add_bounding_box(coordinate, img, [0, 0, 255])

    if tp == 0:
        fn = 1

    precision = round(tp / (tp + fp), 2)
    recall = round(tp / (tp + fn), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, hp.create_info_data(tp, fp, fn, precision, recall)


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box):
    final_name = hp.create_directory_and_get_file_name(main_directory, DIRECTORY_CNN, directory, name, type)
    frameSize = (video.shape[1], video.shape[0])

    out = cv2.VideoWriter(final_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, frameSize, True)

    all_info_data = []

    for i in range(video.shape[3]):
        img = video[:, :, :, i]

        landmark_image = landmarks[:, :, i]
        bounding_box_image = bounding_box[:, :, i]

        img, info_data = find_faces(img, bounding_box_image)
        all_info_data.append(info_data)

        img = hp.add_landmarks(landmark_image, img, [0, 0, 255])
        img = hp.add_bounding_box(bounding_box_image, img, [0, 255, 0])

        out.write(img)

    hp.create_info_file(main_directory, DIRECTORY_CNN, directory, name, type, all_info_data)

    out.release()
