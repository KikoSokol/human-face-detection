import cv2

import save_video as sv


DIRECTORY_VIOLA_JONES = "viola_jones/"


def get_four_vertices(column, row, width, height):
    x0 = [column, row]
    x1 = [column, row + height]
    x2 = [column + width, row]
    x3 = [column + width, row + height]

    return [x0, x1, x2, x3]


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

        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        detected_faces = face_cascade.detectMultiScale(grayscale_image)

        for (column, row, width, height) in detected_faces:
            sur = get_four_vertices(column, row, width, height)
            img = sv.add_bounding_box(sur, img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        landmark_image = landmarks[:, :, i]
        bounding_box_image = bounding_box[:, :, i]

        img = sv.add_landmarks(landmark_image, img)
        img = sv.add_bounding_box(bounding_box_image, img)

        out.write(img)

    out.release()
