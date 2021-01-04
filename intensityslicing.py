# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:23:56 2020

@author: OLLIVANDER
"""

import cv2 as cv
import numpy as np

img = cv.imread("puppy.jpg")
cv.imshow('img', img) 

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_slice = np.zeros(img.shape, dtype='uint8')
shape = img_gray.shape
min_val = 15
max_val = 85

for i in range(shape[0]):
    for j in range(shape[1]):
        if img_gray[i][j]>min_val and img_gray[i][j]<max_val:
            img_slice[i,j] = 255 
        else:
            img_slice[i,j] = 0
cv.imwrite("image_sliced.png", img_slice)

cv.imshow("image", img_slice)
cv.waitKey(0)
cv.destroyAllWindows()
