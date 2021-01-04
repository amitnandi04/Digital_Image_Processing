import cv2 as cv
import numpy as np
img = cv.imread("img1.jpg")

c =2.0
img_log = c * (np.log(img + 1)) 
img_log = np.array(img_log, dtype = np.uint8) 
cv.imwrite("Log Transformation2.png", img_log)

c =3
img_log = c * (np.log(img + 1)) 
img_log = np.array(img_log, dtype = np.uint8) 
cv.imwrite("Log Transformation3.png", img_log)

c =4
img_log = c * (np.log(img + 1)) 
img_log = np.array(img_log, dtype = np.uint8) 
cv.imwrite("Log Transformation4.png", img_log)