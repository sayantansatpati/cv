__author__ = 'ssatpati'

import cv2
import numpy as np

OPENCV_HOME = "/Users/ssatpati/anaconda/pkgs/opencv3-3.1.0-py27_0/share/OpenCV/haarcascades/"

face_cascade = cv2.CascadeClassifier('{0}/haarcascade_frontalface_default.xml'.format(OPENCV_HOME))
eye_cascade = cv2.CascadeClassifier('{0}/haarcascade_eye.xml'.format(OPENCV_HOME))

img = cv2.imread('resources/face2.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print "Gray\n: {0}".format(gray)

faces = face_cascade.detectMultiScale(gray, 1.3, 4)

print "Faces\n: {0}".format(faces)
print "\n### Number of Faces: {0}\n".format(np.shape(faces)[0])

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
