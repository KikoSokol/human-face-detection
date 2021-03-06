import cv2
from mtcnn.mtcnn import MTCNN

import helper as hp

DIRECTORY_DOTS = "dots/"


def find_dots(img, correct_bounding_box, landmarks, detector):
    faces = detector.detect_faces(img)

    eyes_pair_bad = []
    correct_eyes = None

    # Print CNN center of eyes
    for face in faces:
        dots = face["keypoints"]

        for i in hp.create_dot_big_dot(dots["right_eye"]):
            img[i[1]][i[0]] = [48, 88, 247]

        for i in hp.create_dot_big_dot(dots["left_eye"]):
            img[i[1]][i[0]] = [48, 88, 247]

        column, row, width, height = face["box"]
        coordinate = hp.get_four_vertices(column, row, width, height)

        iou = hp.compute_squares_iou(coordinate, correct_bounding_box)

        if iou > 0.5:
            img = hp.add_bounding_box(coordinate, img, [48, 88, 247])
            correct_eyes = (dots["left_eye"], dots["right_eye"])
        else:
            img = hp.add_bounding_box(coordinate, img, [0, 0, 255])
            eyes_pair_bad.append(dots)

    eye_landmarks = hp.get_eye_landmarks(landmarks)

    # Print dot in edges of eyes
    for eye_land in eye_landmarks:
        for dot in hp.create_dot_big_dot(eye_land):
            img[int(dot[1])][int(dot[0])] = [0, 0, 255]

    # Print computed center of eyes
    center_left = hp.get_center_eye(eye_landmarks[0], eye_landmarks[1])
    center_right = hp.get_center_eye(eye_landmarks[2], eye_landmarks[3])

    for i in hp.create_dot_big_dot(center_left):
        img[i[1]][i[0]] = [0, 255, 0]

    for i in hp.create_dot_big_dot(center_right):
        img[i[1]][i[0]] = [0, 255, 0]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Compute distances
    if correct_eyes is None:
        distances = None
    else:
        distances = {"left": hp.distance_points(correct_eyes[0], center_left),
                     "right": hp.distance_points(correct_eyes[1], center_right)}

    return img, correct_eyes, distances, eyes_pair_bad


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box, detector):
    final_name = hp.create_directory_and_get_file_name(main_directory, DIRECTORY_DOTS, directory, name, type)
    frameSize = (video.shape[1], video.shape[0])

    out = cv2.VideoWriter(final_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, frameSize, True)

    left_eye_sum = 0
    right_eye_sum = 0
    eyes_count = 0

    right_distances = {}
    left_distances = {}

    for i in range(video.shape[3]):
        # print(i)
        img = video[:, :, :, i]

        landmark_image = landmarks[:, :, i]
        bounding_box_image = bounding_box[:, :, i]

        img, correct_eyes, distances, eyes_pair_bad = find_dots(img, bounding_box_image, landmark_image, detector)

        if distances is not None:
            eyes_count += 1
            left_eye_sum += distances["left"]
            right_eye_sum += distances["right"]
            left_distances[i] = distances["left"]
            right_distances[i] = distances["right"]

        img = hp.add_landmarks(landmark_image, img, [0, 0, 255])
        img = hp.add_bounding_box(bounding_box_image, img, [0, 255, 0])

        out.write(img)

    print("MSE: " + name + " " + type)
    print("Left eye: " + str(left_eye_sum / eyes_count))
    print("Right eye: " + str(right_eye_sum / eyes_count))

    inverse_right_distances = [(value, key) for key, value in right_distances.items()]
    inverse_left_distances = [(value, key) for key, value in left_distances.items()]

    left_biggest_distance = max(inverse_left_distances)
    right_biggest_distance = max(inverse_right_distances)

    print("Najhor??ie framy:")
    print("left: " + str(left_biggest_distance[1]) + " - " + str(left_biggest_distance[0]))
    print("right: " + str(right_biggest_distance[1]) + " - " + str(right_biggest_distance[0]))

    out.release()
