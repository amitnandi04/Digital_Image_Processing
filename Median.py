import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img=cv.imread('img1.jpg')
median = cv.medianBlur(img,5)
cv.imwrite("Median.png",median)


kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
laplace_sharp = cv.filter2D(img, -1, kernel)
cv.imwrite("Laplace Sharp.png",laplace_sharp)
