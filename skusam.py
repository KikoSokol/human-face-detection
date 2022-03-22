from PIL import Image
import numpy as np

# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# print(data)
# data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
#
# img = Image.fromarray(data, 'RGB')
# img.save('my.png')
# img.show()

##################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

original_image = cv2.imread("1.jpg")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

print(original_image)

grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
detected_faces = face_cascade.detectMultiScale(grayscale_image)
for (column, row, width, height) in detected_faces:
    cv2.rectangle(original_image, (column, row), (column + width, row + height), (0, 255, 0), 4)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
