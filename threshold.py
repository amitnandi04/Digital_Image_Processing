
import cv2 as cv
import numpy as np  

image1 = cv.imread('puppy.jpg') 

img = cv.cvtColor(image1, cv.COLOR_BGR2GRAY) 

ret, thresh1 = cv.threshold(img, 120, 255, cv.THRESH_BINARY) 
ret, thresh2 = cv.threshold(img, 120, 255, cv.THRESH_BINARY_INV) 
ret, thresh3 = cv.threshold(img, 120, 255, cv.THRESH_TRUNC) 
ret, thresh4 = cv.threshold(img, 120, 255, cv.THRESH_TOZERO) 
ret, thresh5 = cv.threshold(img, 120, 255, cv.THRESH_TOZERO_INV) 

cv.imshow('Binary Threshold', thresh1) 
cv.imshow('Binary Threshold Inverted', thresh2) 
cv.imshow('Truncated Threshold', thresh3) 
cv.imshow('Set to 0', thresh4) 
cv.imshow('Set to 0 Inverted', thresh5) 
	
if cv.waitKey(0) & 0xff == 27: 
	cv.destroyAllWindows() 
