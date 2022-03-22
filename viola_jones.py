import cv2

import helper as hp

DIRECTORY_VIOLA_JONES = "viola_jones/"


def find_faces(img):
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)

    coordinates = []
    for (column, row, width, height) in detected_faces:
        coordinate = hp.get_four_vertices(column, row, width, height)
        coordinates.append(coordinate)
        img = hp.add_bounding_box(coordinate, img, [48, 88, 247])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, coordinates


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box):
    final_name = hp.create_directory_and_get_file_name(main_directory, DIRECTORY_VIOLA_JONES, directory, name, type)
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
