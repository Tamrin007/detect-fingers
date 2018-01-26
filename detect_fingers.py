import cv2
import copy
import numpy as np


# Capture stream from webcam
cap = cv2.VideoCapture(0)

# [TODO] Optimize threshold
# See at:
# - https://docs.opencv.org/3.4.0/db/d5c/tutorial_py_bg_subtraction.html
# - https://www.youtube.com/watch?v=T-L9FoH3D9w
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=False)


# if capturing
while True:
    # Read from webcam stream
    ret, frame = cap.read()

    # Apply Background Substraction mask
    fgmask = fgbg.apply(frame)

    # Clip moving object
    img = cv2.bitwise_and(frame, frame, mask=fgmask)

    # Gray scaling
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    # [TODO] Optimize blur size (ksize)
    blur = cv2.GaussianBlur(src=gray, ksize=(21, 21), sigmaX=0)

    # Binarize
    # [TODO] Optimize threshold(thresh)
    ret, binarized = cv2.threshold(src=blur, thresh=60, 255, cv2.THRESH_BINARY)

    # Show frame
    cv2.imshow('frame', binarized)

    # [TODO] detect fingers

    # Quit application
    k = cv2.waitKey(30) & 0xff
    # ESC key pressed
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
