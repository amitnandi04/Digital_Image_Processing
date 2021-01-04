# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:28:32 2020

@author: OLLIVANDER
"""

import cv2 as cv
import numpy as np

img = cv.imread("puppy.jpg")
cv.imshow('img', img) 

xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')
img_contrast = cv.LUT(img, table)