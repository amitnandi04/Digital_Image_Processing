# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:29:30 2020

@author: OLLIVANDER
"""

import cv2 as cv

img = cv.imread("puppy.jpg")
cv.imshow('img', img) 

img_inv = (255 - img)
cv.imwrite("image_negative.png", img_inv)