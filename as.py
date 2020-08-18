from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


image = cv2.imread('/Users/hemnathraja/Indrasol/Cam_scan_demo/d.jpeg')
#ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 600)
image = image.astype('uint8')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(image, (5,3), 0)
edged = cv2.Canny(gray, 75, 250)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
#dilated = cv2.dilate(image, kernel)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

cv2.imwrite("Image.jpg", image)
cv2.imwrite("Edged.jpg", edged)
cv2.imwrite("approx.jpg", image)




    #print(len(approx))

cv2.imshow("approx", image)
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)


cv2.waitKey(0)
cv2.destroyAllWindows()
