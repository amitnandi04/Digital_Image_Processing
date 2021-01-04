# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:25:37 2020

@author: OLLIVANDER
"""

import cv2 as cv
import numpy as np

img = cv.imread("puppy.jpg")
cv.imshow('img', img) 

img_log = ( 255 / np.log(1 + np.max(img)) )* np.log(img + 1)
img_log = np.array(img_log, dtype = np.uint8) 
cv.imwrite("image_log.png", img_log)