import cv2
from fer import FER
import matplotlib.pyplot as plt


video_path = 'C:/Users/never/Desktop/activity-recognition/video_1.mp4'
detector = FER(mtcnn=True)
# cap = cv2.VideoCapture(video_path)  # Use the video file instead of the camera (0)
cap = cv2.VideoCapture(0) #using real time stream

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    emotions = detector.detect_emotions(frame)

    for emotion, score in emotions[0]["emotions"].items():
        print(f"{emotion}: {score}")

    bounding_box = emotions[0]["box"]
    # get the best detected emotion from emotions list
    emotion_text = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)

    cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 0, 255), 2)
    
    # Display emotion text on the frame
    cv2.putText(frame, emotion_text, (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # display_frame(frame)
    cv2.imshow('video frames', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
