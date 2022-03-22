import cv2

import save_video as sv


DIRECTORY_VIOLA_JONES = "viola_jones/"


def get_four_vertices(column, row, width, height):
    x0 = [column, row]
    x1 = [column, row + height]
    x2 = [column + width, row]
    x3 = [column + width, row + height]

    return [x0, x1, x2, x3]


def find_faces(img):
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)

    coordinates = []
    for (column, row, width, height) in detected_faces:
        coordinate = get_four_vertices(column, row, width, height)
        coordinates.append(coordinate)
        img = sv.add_bounding_box(coordinate, img, [48, 88, 247])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, coordinates


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box):
    frameSize = (video.shape[1], video.shape[0])

    dir = main_directory + DIRECTORY_VIOLA_JONES
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
