import cv2
from fer import FER
import matplotlib.pyplot as plt
import numpy as np


# image path
image_path = 'images_folder/download (4).jfif'

# passing the image to opencv 
image = cv2.imread(image_path)

# face detection model is set to mtcnn
detector = FER(mtcnn=True)

# pass the image from emotion detector
emotions = detector.detect_emotions(image)

for emotion, score in emotions[0]["emotions"].items():
    print(f"{emotion}: {score}")

# setting the detection box in display
bounding_box = emotions[0]["box"]

# visualization of detection box in display
cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 0, 255), 2)

# Display the image 
cv2.imshow('image', image)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
