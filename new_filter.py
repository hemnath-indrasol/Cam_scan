#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 02:14:10 2020

@author: hemnathraja
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
#import pandas as pd
from ocr.helpers import implt, resize, ratio

%matplotlib inline
plt.rcParams['figure.figsize'] = (9.0, 9.0)

image = cv2.cvtColor(cv2.imread("/Users/hemnathraja/Indrasol//Cam_scan_demo/demo.jpeg"), cv2.COLOR_BGR2RGB)
implt(image)


def edges_det(img, min_val, max_val):
    img = cv2.cvtColor(resize(img), cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    implt(img, 'gray', 'Adaptive Threshold')
    
    img = cv2.medianBlur(img, 11)
    
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value =[0, 0, 0])
    implt(img, 'gray', 'Median Blur + Border')
    
    return cv2.Canny(img, min_val, max_val)

edges_image = edges_det(image, 200, 250)

edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))
implt(edges_image, 'gray', 'Edges')
   

def four_corners_sort(pts):
    diff = np.diff(pts, axis =1)
    summ = pts.sum(axis = 1)
    
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contour_offset(cnt, offset):
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt
    
    
def find_page_contours(edges, img):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_CONTOUR_AREA = height * width * 0.5
    MAX_CONTOUR_AREA = (width - 10) * (height - 10)

    max_area = MIN_CONTOUR_AREA

    page_contour = np.array([[0,0], [0, height-5], [width-5, height-5], [width-5,0]])
    
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
    
        if (len(approx) == 4 and cv2.isContourConvex(approx) and max_area < cv2.contourArea(approx) < MAX_CONTOUR_AREA):
            max_area = cv2.contourArea(approx)
            page_contour = approx[:, 0]
            
    page_contour = four_corner_sort(page_contour)
    return contour_offset(page_contour, (-5, -5))


page_contour = find_page_contours(edges_image, resize(image))


print("Page Contour")
print(page_contour)
implt(cv2.drawContours(resize(image), [page_contour], -1, (0, 255, 0), 3))


page_contour = page_contour.dot(ratio(image))



def persp_transform(img, s_points):
    height = max(np.linalg.norm(s_points[0] - s_points[1]), np.linalg.norm(s_points[2] - s_points[3]))
    
    width = max(np.linalg.norm(s_points[1] - s_points[2]), np.linalg.norm(s_points[3] - s_points[0]))
    
    t_points = np.array([[0,0], [0, height], [width, height], [width, 0]], np.float32)
    
    if s_points.dtype != np.float32:
        s_points = s_points.astype(np.float32)
    
    M = cv2.getPerspectiveTransform(s_points, t_points)
    
    return cv2.warpPerspective(img, M, (int(width), int(height)))

newImage = persp_transform(image, page_contour)

implt(newImage, t = "Result")
    
    
    




#cv2.imwrite("/Users/hemnathraja/Indrasol//Cam_scan_demo/Output/new.jpg", cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    