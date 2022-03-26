import cv2
from mtcnn.mtcnn import MTCNN

import helper as hp

DIRECTORY_DOTS = "dots/"


def find_dots(img, landmarks, detector):
    faces = detector.detect_faces(img)

    eyes_pair = []

    # Print CNN center of eyes
    for face in faces:
        dots = face["keypoints"]

        for i in hp.create_dot_big_dot(dots["right_eye"]):
            img[i[1]][i[0]] = [48, 88, 247]

        for i in hp.create_dot_big_dot(dots["left_eye"]):
            img[i[1]][i[0]] = [48, 88, 247]

        eyes_pair.append((dots["left_eye"], dots["right_eye"]))

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

    return img


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box, detector):
    final_name = hp.create_directory_and_get_file_name(main_directory, DIRECTORY_DOTS, directory, name, type)
    frameSize = (video.shape[1], video.shape[0])

    out = cv2.VideoWriter(final_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, frameSize, True)

    for i in range(video.shape[3]):
        print(i)
        img = video[:, :, :, i]

        landmark_image = landmarks[:, :, i]
        bounding_box_image = bounding_box[:, :, i]

        img = find_dots(img, landmark_image, detector)

        img = hp.add_landmarks(landmark_image, img, [0, 0, 255])
        img = hp.add_bounding_box(bounding_box_image, img, [0, 255, 0])

        out.write(img)

    out.release()
