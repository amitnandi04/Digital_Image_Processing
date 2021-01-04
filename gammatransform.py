# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:30:34 2020

@author: OLLIVANDER
"""

import cv2 as cv
import numpy as np

img = cv.imread("puppy.jpg")

gamma = 10
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
img_gamma = cv.LUT(img, lookUpTable)
cv.imwrite("image_gamma.png", img_gamma)
