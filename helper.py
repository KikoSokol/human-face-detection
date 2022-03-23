import os

import numpy as np


def add_bounding_box(bounding_box, image, color):
    for up in range(int(bounding_box[0][1]), int(bounding_box[1][1]) + 1):
        image[up][int(bounding_box[0][0])] = np.array(color)

    for down in range(int(bounding_box[2][1]), int(bounding_box[3][1]) + 1):
        image[down][int(bounding_box[2][0])] = np.array(color)

    for left in range(int(bounding_box[0][0]), int(bounding_box[2][0]) + 1):
        image[int(bounding_box[0][1])][left] = np.array(color)

    for right in range(int(bounding_box[1][0]), int(bounding_box[3][0]) + 1):
        image[int(bounding_box[1][1])][right] = np.array(color)

    return image


def add_landmarks(landmarks, image, color):
    for i in landmarks:
        image[int(i[1])][int(i[0])] = np.array(color)

    return image


def create_folder(name):
    if os.path.isdir(name) is True:
        return None
    else:
        os.mkdir(name[0: len(name) - 1])


def get_four_vertices(column, row, width, height):
    x0 = [column, row]
    x1 = [column, row + height]
    x2 = [column + width, row]
    x3 = [column + width, row + height]

    return [x0, x1, x2, x3]


def compute_iou(found_face_square, correct_square):
    x_inter1 = max(found_face_square[0], correct_square[0])
    y_inter1 = max(found_face_square[1], correct_square[1])
    x_inter_2 = min(found_face_square[2], correct_square[2])
    y_inter_2 = min(found_face_square[3], correct_square[3])

    width_inter = abs(x_inter_2 - x_inter1)
    height_inter = abs(y_inter_2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(found_face_square[2] - found_face_square[0])
    height_box1 = abs(found_face_square[3] - found_face_square[1])
    width_box2 = abs(correct_square[2] - correct_square[0])
    height_box2 = abs(correct_square[3] - correct_square[1])

    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter

    return area_inter / area_union


def get_cord(square_vertices):
    return square_vertices[0][1], square_vertices[0][0], square_vertices[3][1], square_vertices[3][0]


def compute_squares_iou(vertices_found_face_square, vertices_correct_square):
    return compute_iou(get_cord(vertices_found_face_square), get_cord(vertices_correct_square))


def create_info_data(tp, fp, fn, precision, recall):
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def create_directory_and_get_file_name(main_directory, category_directory, directory, name, type):
    dir = main_directory + category_directory
    create_folder(dir)

    directory = dir + directory
    create_folder(directory)

    return directory + name + "_" + type + ".mp4"
