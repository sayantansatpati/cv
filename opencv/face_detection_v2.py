__author__ = 'ssatpati'

import cv2
import numpy as np

OPENCV_HOME = "/Users/ssatpati/anaconda/pkgs/opencv3-3.1.0-py27_0/share/OpenCV/haarcascades/"

face_cascade = cv2.CascadeClassifier('{0}/haarcascade_frontalface_default.xml'.format(OPENCV_HOME))
eye_cascade = cv2.CascadeClassifier('{0}/haarcascade_eye.xml'.format(OPENCV_HOME))


def detect_faces(video_file):
    cap = cv2.VideoCapture(video_file)

    total_faces = 0

    while cap.isOpened():
        try:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #print "Gray\n: {0}".format(gray)

            faces = face_cascade.detectMultiScale(gray, 1.3, 4)

            print "Faces\n: {0}".format(faces)
            print "\n### Number of Faces: {0}\n".format(np.shape(faces)[0])
            total_faces += np.shape(faces)[0]

            '''
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            '''
        except Exception as e:
            print e
            break

    cap.release()
    cv2.destroyAllWindows()

    return total_faces

if __name__ == '__main__':
    '''Main Point of Entry to Program'''
    VIDEO_FILE = 'resources/video.avi'
    print "\n### Total Number of Faces Detected: {0}".format(detect_faces(VIDEO_FILE))