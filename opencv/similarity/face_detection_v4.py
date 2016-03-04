__author__ = 'ssatpati'

import cv2
import numpy as np
import sys
import datetime

OPENCV_HOME = "/Users/ssatpati/anaconda/pkgs/opencv3-3.1.0-py27_0/share/OpenCV/haarcascades/"

face_cascade = cv2.CascadeClassifier('{0}/haarcascade_frontalface_default.xml'.format(OPENCV_HOME))
eye_cascade = cv2.CascadeClassifier('{0}/haarcascade_eye.xml'.format(OPENCV_HOME))

#Set the RATIO by which you want to resize the image.
#Based on the video quality decide the
RATIO = 2
#Enter the number of frames you want to track the face after detection
TRACK = 30
#Enter the number of frames to be skipped if no face is found
SKIP = 2
#Load the video. Enter the location of video on which you want to apply
cap = cv2.VideoCapture('../resources/video.avi')
# initialize the termination criteria for CAMSHIFT, indicating
# a maximum of ten iteRATIOns or movement by a least one pixel
# along with the bounding box of the ROI
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

TOTAL_FACES_DETECTED = 0
TOTAL_ESTIMATED_UNIQ_FACES = 0


def detect_faces(frame_id, frame):
    """Detect Faces"""
    global TOTAL_FACES_DETECTED

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = face_cascade.detectMultiScale(gray, 1.3, 4)

    TOTAL_FACES_DETECTED += np.shape(faces_rects)[0]
    print "\n### [{0}] Faces Detected: {1}\n{2}\n".format(frame_id, np.shape(faces_rects)[0], faces_rects)

    for (x, y, w, h) in faces_rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    return faces_rects

# Assuming video snippets won't contain more than a few face ROI(s)
def get_uniq_faces_curr_frame(frame_id, faces_roi_hists_prev, faces_roi_hists):
    print "\n### [{0}] Face Similarity: Prev: {1}, Curr: {2}".format(frame_id,
                                                                     len(faces_roi_hists_prev),
                                                                     len(faces_roi_hists))
    # First Time
    if len(faces_roi_hists_prev) == 0:
        return len(faces_roi_hists)

    uniq_faces_curr_frame = 0

    # Perform Image Histogram Similarity
    # For each histogram in current frame
    for rh1 in faces_roi_hists:
        match_found = False
        # For each histogram in previous frame
        for rh2 in faces_roi_hists_prev:
            #print "\nrh1 {0}: {1}".format(type(rh1),np.shape(rh1))
            #print "\nrh2 {0}: {1}".format(type(rh2),np.shape(rh2))
            corr = cv2.compareHist(rh1, rh2, cv2.HISTCMP_CORREL)
            print "### [{0}] Similarity Metrics: {1}".format(frame_id, corr)
            if corr >= 0.40:
                # Match Found, can break the loop for this histogram in current frame
                match_found = True
                break;
        # Add to unique face count, if no match found for this histogram in current frame
        if not match_found:
            uniq_faces_curr_frame += 1

    print "### [{0}] Total Unique Faces in Current Frame: {1}\n".format(frame_id, uniq_faces_curr_frame)
    return uniq_faces_curr_frame


def get_roi_hist(faces_rects):
    faces_roi_hists = []
    for (x, y, w, h) in faces_rects:
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0], mask, [180], [0,180])
        roi_hist = cv2.normalize(roi_hist,roi_hist, 0, 255, cv2.NORM_MINMAX)
        faces_roi_hists.append(roi_hist)
    return faces_roi_hists


if __name__ == '__main__':
    '''Main Point of Entry to Program'''
    start = datetime.datetime.now()

    global TOTAL_FACES_DETECTED
    global TOTAL_ESTIMATED_UNIQ_FACES

    VIDEO_FILE = '../resources/video.avi'

    cap = cv2.VideoCapture(VIDEO_FILE)

    # Hists from Prev Frames
    faces_roi_hists_prev = []

    frame_id = 0
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            frame_id += 1

            # Get the Face ROI from Viola Jones
            faces_rects = detect_faces(frame_id, frame)

            # If Faces Detected
            if np.shape(faces_rects)[0] > 0:
                # Get the Histograms for the ROI(s) detected
                faces_roi_hists = get_roi_hist(faces_rects)

                # Detect Similar Faces
                TOTAL_ESTIMATED_UNIQ_FACES += get_uniq_faces_curr_frame(frame_id, faces_roi_hists_prev, faces_roi_hists)

                # Set Previous
                faces_roi_hists_prev = faces_roi_hists

            # Show the Frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Count
            print "\n### [{0}] Total Estimated Unique Faces so far: {1}".format(frame_id, TOTAL_ESTIMATED_UNIQ_FACES)

            # For Debugging Purposes
            #if frame_id > 300:
            #    break
        except Exception as e:
            print e
            break

    cap.release()
    cv2.destroyAllWindows()

    print "\n\n################### REPORT ###################"
    print "### Total Number of Faces Detected: {0} ###".format(TOTAL_FACES_DETECTED)
    print "### Total Estimated Unique Faces: {0} ###".format(TOTAL_ESTIMATED_UNIQ_FACES)
    print "### Total acquisition time: " + str(datetime.datetime.now() - start) + "###"
    print "\n\n################### ###### ###################"
