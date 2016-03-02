__author__ = 'ssatpati'

import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Show Resized Frame Frames
    #cv2.imshow('Input', frame)

    # Show Edge Detection Frames
    cv2.imshow('Canny', cv2.Canny(frame, 50, 240))

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
