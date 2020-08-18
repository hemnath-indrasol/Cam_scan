#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


img = cv2.imread('/Users/hemnathraja/Indrasol/Cam_scan_demo/demo_txt.jpeg')


# In[3]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[4]:


blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


# In[9]:


#Apply morphology
kernel = np.ones((7,7), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)


# In[11]:


#Get largest Contour
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
area_thresh = 0
for c in contours:
    area = cv2.contourArea(c)
    if area > area_thresh:
        area_thresh = area
        big_contour = c
        
#Draw 
page = np.zeros_like(img)
cv2.drawContours(page, [big_contour], 0, (255,255,255), -1)

#Get perimeter and ~polygon
peri = cv2.arcLength(big_contour, True)
corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)

#Draw polygon on input img from detedcted corner
polygon = img.copy()
cv2.polylines(polygon, [corners], True, (0,0,255), 1, cv2.LINE_AA)

print(len(corners))
print(corners)


width = 0.5 * ((corners[0][0][0] - corners[1][0][0]) + (corners[3][0][0] - corners[2][0][0]))
height = 0.5 * ((corners[2][0][1] - corners[1][0][1]) + (corners[3][0][1] - corners[0][0][1]))

width = np.int0(width)
height = np.int0(height)


# In[12]:


#Reform input corners to x,y list
icorners = []
for corner in corners:
    pt = [corner[0][0], corner[0][1]]
    icorners.append(pt)
icorners = np.float32(icorners)

#Get corresponding output corners from width and height
ocorners = [[width,0], [0,0], [0,height], [width,height]]
ocorners = np.float32(ocorners)

#Get perspective transform matrix
M = cv2.getPerspectiveTransform(icorners, ocorners)

warped = cv2.warpPerspective(img, M, (width, height))


# In[13]:


#Write Results
cv2.imwrite("efile_thresh.jpg", thresh)
cv2.imwrite("efile_morph.jpg", morph)
cv2.imwrite("efile_polygon.jpg", polygon)
cv2.imwrite("efile_warped.jpg", warped)


# In[ ]:


#Display it
cv2.imshow("efile_thresh", thresh)
cv2.imshow("efile_morph", morph)
cv2.imshow("efile_page", page)
cv2.imshow("efile_polygon", polygon)
cv2.imshow("efile_warped", warped)
cv2.waitKey(0)


# In[ ]:




