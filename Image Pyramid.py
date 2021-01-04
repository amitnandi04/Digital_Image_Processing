
import cv2

img = cv2.imread('img1.jpg')
h, w, _channnel = img.shape
lower_reso = cv2.pyrDown(img, dstsize=(w//2, h//2))
higher_reso = cv2.pyrUp(img, dstsize=(w*2, h*2))
cv2.imshow("lower reso", lower_reso)
cv2.imwrite("image pyramid lower reso.png", lower_reso)
cv2.imwrite("image pyramid higher reso.png", higher_reso)
