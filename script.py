import cv2
import numpy as np


#EDGE DETECTION
image = cv2.imread('mango37.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blurred, 30, 150)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9, 9))
dilated = cv2.dilate(canny, kernel)
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
result = cv2.bitwise_and(image, image, mask=closing)
image = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
closing = cv2.resize(closing, (0, 0), fx = 0.5, fy = 0.5)
cv2.imshow("Original", image)
cv2.imshow("Canny", closing)
result[np.where((result==[0, 0, 0]).all(axis=2))] = [255, 255, 255]
result[dilated == 0] = 255
result[dilated == 0] = 255
result[dilated == 255] = 255
result = cv2.resize(result, (0, 0), fx = 0.5, fy = 0.5)
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (3, 3),0)
blurred = cv2.medianBlur(blurred,1)
gray = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)

canny = cv2.Canny(blurred, 30, 150)
canny = cv2.resize(gray, (0, 0), fx = 2, fy = 2)
cv2.imshow("res", canny)
cv2.waitKey(0)



'''


BACKGROUND REMOVAL

img = cv2.imread('mango1.jpg')
screen_res = 1280, 720
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)
#resized window width and height



original = img.copy()
image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 0], dtype="uint8")
upper = np.array([255, 255, 100], dtype="uint8")
mask = cv2.inRange(image, lower, upper)
mask=~mask
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(mask, cnts, (255,255,255))
result = cv2.bitwise_and(original,original,mask=mask)
result[mask < 255] = 255
result = cv2.resize(result, (0, 0), fx = 1, fy = 1)
kernel = np.ones((2,2),np.uint8)
dilation = cv2.dilate(result,kernel,iterations = 2)
window_width = int(dilation.shape[1] * scale)
window_height = int(dilation.shape[0] * scale)
cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resized Window', window_width, window_height)
cv2.imshow('Resized Window',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

-----------------------------------------------------------------

import cv2
import numpy as np

img = cv2.imread('mango3.jpg')
img=cv2.resize(img, (960, 540)) 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("mask.png", thresh)
mask = cv2.imread('mask.png')

# you can directly use thresh also instead of saving and loading again

result = cv2.bitwise_and(img, mask)
result[mask==0] = 255

cv2.imshow('image', img)
cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey()
cv2.destroyAllWindows()'''


# SECOND WAY
