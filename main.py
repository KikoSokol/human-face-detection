import numpy as np
import os

import save_video as sv
import viola_jones as vj
import cnn
import all
import search_faces_dots as ds
import helper as hp
from mtcnn.mtcnn import MTCNN

FOLDER_WITH_NPZ = "videa/"
FOLDER_WITH_MP4 = "mp4_porovnanie/"

npz_files = []

trainDirectory = 'videa/'


for filename in os.listdir(trainDirectory):
    npz_files.append(filename)

npz_files = [npz_files[0]]

print(npz_files)

###########################ALL###################################################

detector = MTCNN()
for file_name in npz_files:
    print(file_name)
    video_file = np.load(FOLDER_WITH_NPZ + file_name)
    file_name_without_suffix = file_name.split(".")[0]
    directory_name = file_name_without_suffix + "/"
    original = all.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "ORIGINAL",
                          video_file["colorImages"],
                          video_file["landmarks2D"],
                          video_file["boundingBox"], detector)

    summary_info = [original]
    hp.create_summary_info(FOLDER_WITH_MP4, "ALL-WITHOUT-VIDEO/", directory_name, file_name_without_suffix + "_SUMMARY",
                           "", summary_info)
