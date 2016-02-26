__author__ = 'ssatpati'

import cv2
import numpy as np

# Load an color image in grayscale
img = cv2.imread('messi5.jpg',0)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
