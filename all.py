import cv2

import helper as hp

DIRECTORY_ALL = "ALL-WITHOUT-VIDEO/"


def find_faces_viola(img, detected_faces, correct_bounding_box):
    tp = 0
    fp = 0
    fn = 0

    for (column, row, width, height) in detected_faces:
        coordinate = hp.get_four_vertices(column, row, width, height)

        iou = hp.compute_squares_iou(coordinate, correct_bounding_box)

        if iou > 0.5:
            tp += 1
            # img = hp.add_bounding_box(coordinate, img, [247, 88, 48])
        else:
            fp += 1
            # img = hp.add_bounding_box(coordinate, img, [0, 0, 255])

    if tp == 0:
        fn = 1

    if tp == 0 and fp == 0:
        precision = 0.0
    else:
        precision = round(tp / (tp + fp), 2)
    recall = round(tp / (tp + fn), 2)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, hp.create_info_data(tp, fp, fn, precision, recall)


def find_faces_cnn(img, faces, correct_bounding_box, landmarks):
    eyes_pair_bad = []
    correct_eyes = None

    tp = 0
    fp = 0
    fn = 0

    # Print CNN center of eyes
    for face in faces:
        dots = face["keypoints"]

        # for i in hp.create_dot_big_dot(dots["right_eye"]):
        #     img[i[1]][i[0]] = [48, 88, 247]
        #
        # for i in hp.create_dot_big_dot(dots["left_eye"]):
        #     img[i[1]][i[0]] = [48, 88, 247]

        column, row, width, height = face["box"]
        coordinate = hp.get_four_vertices(column, row, width, height)

        iou = hp.compute_squares_iou(coordinate, correct_bounding_box)

        if iou > 0.5:
            tp += 1
            # img = hp.add_bounding_box(coordinate, img, [48, 88, 247])
            correct_eyes = (dots["left_eye"], dots["right_eye"])
        else:
            fp += 1
            # img = hp.add_bounding_box(coordinate, img, [0, 0, 255])
            eyes_pair_bad.append(dots)

    if tp == 0:
        fn = 1

    if tp == 0 and fp == 0:
        precision = 0.0
    else:
        precision = round(tp / (tp + fp), 2)
    recall = round(tp / (tp + fn), 2)

    eye_landmarks = hp.get_eye_landmarks(landmarks)

    # Print dot in edges of eyes
    # for eye_land in eye_landmarks:
    #     for dot in hp.create_dot_big_dot(eye_land):
    #         img[int(dot[1])][int(dot[0])] = [0, 0, 255]

    # Print computed center of eyes
    center_left = hp.get_center_eye(eye_landmarks[0], eye_landmarks[1])
    center_right = hp.get_center_eye(eye_landmarks[2], eye_landmarks[3])

    # for i in hp.create_dot_big_dot(center_left):
    #     img[i[1]][i[0]] = [0, 255, 0]
    #
    # for i in hp.create_dot_big_dot(center_right):
    #     img[i[1]][i[0]] = [0, 255, 0]

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Compute distances
    if correct_eyes is None:
        distances = None
    else:
        distances = {"left": hp.distance_points(correct_eyes[0], center_left),
                     "right": hp.distance_points(correct_eyes[1], center_right)}

    return img, correct_eyes, distances, eyes_pair_bad, hp.create_info_data(tp, fp, fn, precision, recall)


def detect_viola(img):
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
    return detected_faces


def detect_cnn(img, detector):
    return detector.detect_faces(img)


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box, detector):
    final_name = hp.create_directory_and_get_file_name(main_directory, DIRECTORY_ALL, directory, name, type)
    # frameSize = (video.shape[1], video.shape[0])

    # out = cv2.VideoWriter(final_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, frameSize, True)

    all_info_data_viola = []
    all_info_data_cnn = []

    left_eye_sum = 0
    right_eye_sum = 0
    eyes_count = 0

    right_distances = {}
    left_distances = {}

    all_viola_precision = []
    all_viola_recall = []

    all_viola_fp = []
    all_viola_fn = []

    all_cnn_precision = []
    all_cnn_recall = []

    all_cnn_fp = []
    all_cnn_fn = []

    for i in range(video.shape[3]):
        img = video[:, :, :, i]

        landmark_image = landmarks[:, :, i]
        bounding_box_image = bounding_box[:, :, i]

        faces_viola = detect_viola(img)
        faces_cnn = detect_cnn(img, detector)

        img, info_data_viola = find_faces_viola(img, faces_viola, bounding_box_image)
        all_info_data_viola.append(info_data_viola)

        all_viola_precision.append(info_data_viola[3])
        all_viola_recall.append(info_data_viola[4])

        all_viola_fp.append(info_data_viola[1])
        all_viola_fn.append(info_data_viola[2])

        img, correct_eyes, distances, eyes_pair_bad, info_data_cnn = find_faces_cnn(img, faces_cnn, bounding_box_image,
                                                                                    landmark_image)
        all_info_data_cnn.append(info_data_cnn)

        all_cnn_precision.append(info_data_cnn[3])
        all_cnn_recall.append(info_data_cnn[4])

        all_cnn_fp.append(info_data_cnn[1])
        all_cnn_fn.append(info_data_cnn[2])

        if distances is not None:
            eyes_count += 1
            left_eye_sum += distances["left"]
            right_eye_sum += distances["right"]
            left_distances[i] = distances["left"]
            right_distances[i] = distances["right"]

        # img = hp.add_landmarks(landmark_image, img, [0, 0, 255])
        # img = hp.add_bounding_box(bounding_box_image, img, [0, 255, 0])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # out.write(img)

    hp.create_info_file(main_directory, DIRECTORY_ALL, directory, name, type + "_viola", all_info_data_viola)
    hp.create_info_file(main_directory, DIRECTORY_ALL, directory, name, type + "_cnn", all_info_data_cnn)

    # print("MSE: " + name + " " + type)
    # print("Left eye: " + str(left_eye_sum / eyes_count))
    # print("Right eye: " + str(right_eye_sum / eyes_count))

    inverse_right_distances = [(value, key) for key, value in right_distances.items()]
    inverse_left_distances = [(value, key) for key, value in left_distances.items()]

    left_biggest_distance = max(inverse_left_distances)
    right_biggest_distance = max(inverse_right_distances)

    # print("Najhoršie framy:")
    # print("left: " + str(left_biggest_distance[1]) + " - " + str(left_biggest_distance[0]))
    # print("right: " + str(right_biggest_distance[1]) + " - " + str(right_biggest_distance[0]))

    # print("Viola Precision:")
    viola_sum_precision = 0
    for i in all_viola_precision:
        viola_sum_precision += i
    # print(viola_sum_precision)

    # print("Viola recall:")
    viola_sum_recall = 0
    for i in all_viola_recall:
        viola_sum_recall += i
    # print(viola_sum_recall)

    # print("CNN Precision:")
    cnn_sum_precision = 0
    for i in all_cnn_precision:
        cnn_sum_precision += i
    # print(cnn_sum_precision)

    # print("CNN recall:")
    cnn_sum_recall = 0
    for i in all_cnn_recall:
        cnn_sum_recall += i
    # print(cnn_sum_recall)

    viola_sum_fp = 0
    for i in all_viola_fp:
        viola_sum_fp += i

    viola_sum_fn = 0
    for i in all_viola_fn:
        viola_sum_fn += i

    cnn_sum_fp = 0
    for i in all_cnn_fp:
        cnn_sum_fp += i

    cnn_sum_fn = 0
    for i in all_cnn_fn:
        cnn_sum_fn += i

    result = [type, viola_sum_precision, viola_sum_precision/len(all_viola_precision),
              viola_sum_recall, viola_sum_recall/len(all_viola_recall),
              viola_sum_fp, viola_sum_fp/len(all_viola_fp), viola_sum_fn, viola_sum_fn/len(all_viola_fn),
              cnn_sum_precision, cnn_sum_precision/len(all_cnn_precision),
              cnn_sum_recall, cnn_sum_recall/len(all_cnn_recall),
              cnn_sum_fp, cnn_sum_fp/len(all_cnn_fp), cnn_sum_fn, cnn_sum_fn/len(all_cnn_fn),
              left_eye_sum / eyes_count, right_eye_sum / eyes_count,
              left_biggest_distance[1], left_biggest_distance[0], right_biggest_distance[1], right_biggest_distance[0]]

    # out.release()

    return result
