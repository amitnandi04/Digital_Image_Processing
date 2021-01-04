# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:38:22 2020

@author: OLLIVANDER
"""

import cv2 as cv
import numpy as np

img = cv.imread("puppy.jpg")
cv.imshow('img', img) 

kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
laplace_sharp = cv.filter2D(img, -1, kernel)
cv.imwrite("laplace_sharp.png", laplace_sharp)