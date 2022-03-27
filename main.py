import numpy as np

import save_video as sv
import viola_jones as vj
import cnn
import all
import search_faces_dots as ds
import helper as hp
from mtcnn.mtcnn import MTCNN

FOLDER_WITH_NPZ = "viz_vzorka/"
FOLDER_WITH_MP4 = "mp4/"

npz_files = ["Kieran_Culkin_0.npz", "Liu_Ye_2.npz", "Maggie_Smith_3.npz", "Margaret_Thatcher_5.npz",
             "Marisa_Tomei_1.npz", "Martin_Sheen_3.npz", "Martin_Sheen_5.npz", "Matt_Anderson_2.npz",
             "Natalie_Stewart_2.npz", "Oscar_Elias_Biscet_0.npz"]

# npz_files = ["Kieran_Culkin_0.npz"]

# for file_name in npz_files:
#     video_file = np.load(FOLDER_WITH_NPZ + file_name)
#     file_name_without_suffix = file_name.split(".")[0]
#     directory_name = file_name_without_suffix + "/"
#     sv.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "ORIGINAL", video_file["colorImages_original"],
#               video_file["landmarks2D"],
#               video_file["boundingBox"])
#     sv.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "MEDIUM", video_file["colorImages_medium"],
#               video_file["landmarks2D"],
#               video_file["boundingBox"])
#     sv.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "SEVERE", video_file["colorImages_severe"],
#               video_file["landmarks2D"],
#               video_file["boundingBox"])

###########################Viola-Jones###################################################
# for file_name in npz_files:
#     video_file = np.load(FOLDER_WITH_NPZ + file_name)
#     file_name_without_suffix = file_name.split(".")[0]
#     directory_name = file_name_without_suffix + "/"
#     vj.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "ORIGINAL", video_file["colorImages_original"],
#               video_file["landmarks2D"],
#               video_file["boundingBox"])
#     vj.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "MEDIUM", video_file["colorImages_medium"],
#               video_file["landmarks2D"],
#               video_file["boundingBox"])
#     vj.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "SEVERE", video_file["colorImages_severe"],
#               video_file["landmarks2D"],
#               video_file["boundingBox"])

###########################CNN###################################################
# for file_name in npz_files:
#     video_file = np.load(FOLDER_WITH_NPZ + file_name)
#     file_name_without_suffix = file_name.split(".")[0]
#     directory_name = file_name_without_suffix + "/"
#     cnn.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "ORIGINAL",
#                video_file["colorImages_original"],
#                video_file["landmarks2D"],
#                video_file["boundingBox"])
#     cnn.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "MEDIUM", video_file["colorImages_medium"],
#                video_file["landmarks2D"],
#                video_file["boundingBox"])
#     cnn.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "SEVERE", video_file["colorImages_severe"],
#                video_file["landmarks2D"],
#                video_file["boundingBox"])

###########################DOTS###################################################
# detector = MTCNN()
# for file_name in npz_files:
#     video_file = np.load(FOLDER_WITH_NPZ + file_name)
#     file_name_without_suffix = file_name.split(".")[0]
#     directory_name = file_name_without_suffix + "/"
#     ds.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "ORIGINAL",
#               video_file["colorImages_original"],
#               video_file["landmarks2D"],
#               video_file["boundingBox"], detector)


###########################ALL###################################################

detector = MTCNN()
for file_name in npz_files:
    video_file = np.load(FOLDER_WITH_NPZ + file_name)
    file_name_without_suffix = file_name.split(".")[0]
    directory_name = file_name_without_suffix + "/"
    original = all.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "ORIGINAL",
                          video_file["colorImages_original"],
                          video_file["landmarks2D"],
                          video_file["boundingBox"], detector)
    medium = all.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "MEDIUM",
                        video_file["colorImages_medium"],
                        video_file["landmarks2D"],
                        video_file["boundingBox"], detector)
    severe = all.to_mp4(FOLDER_WITH_MP4, directory_name, file_name_without_suffix, "SEVERE",
                        video_file["colorImages_severe"],
                        video_file["landmarks2D"],
                        video_file["boundingBox"], detector)

    summary_info = [original, medium, severe]
    hp.create_summary_info(FOLDER_WITH_MP4, "ALL/", directory_name, "SUMMARY", "", summary_info)
