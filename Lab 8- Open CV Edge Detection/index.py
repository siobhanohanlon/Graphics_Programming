#Lab 8

# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Variables
rows = 4
cols = 2
KernelS1 = 5
KernelS2 = 13

# Import Image
img = cv2.imread('ATU.jpg',)
#Own Image
#img = cv2.imread('Beach.jpg')

# GreyScale Image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Plot Images one by one
cv2.imshow('Original image', img)
cv2.imshow('Grey image', gray)

# Wait and Close Windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Blur Images
imgOut = cv2.GaussianBlur(gray, (KernelS1, KernelS1), 0)
imgOut2 = cv2.GaussianBlur(gray, (KernelS2, KernelS2), 0)

# Sobel- Edge Detection
sobelH = cv2.Sobel(imgOut, cv2.CV_64F, 1, 0, ksize=5)  # x dir
sobelV = cv2.Sobel(imgOut, cv2.CV_64F, 0, 1, ksize=5)  # y dir
# sobelSum = sobelH + sobelV
sobelSum = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5) + \
    cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# Canny
canny = cv2.Canny(imgOut, 100, 200)

# Plot Multiple Images
plt.subplot(rows, cols, 1), plt.imshow(
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 2), plt.imshow(gray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 3), plt.imshow(imgOut, cmap='gray')
plt.title('5x5 Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 4), plt.imshow(imgOut2, cmap='gray')
plt.title('13x13 Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 5), plt.imshow(sobelH, cmap='gray')
plt.title('Sobel Horizontal'), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 6), plt.imshow(sobelV, cmap='gray')
plt.title('Sobel Vertical'), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 7), plt.imshow(sobelSum, cmap='gray')
plt.title('Sobel Sum'), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 8), plt.imshow(canny, cmap='gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])

plt.show()

# For loops over 5 times and changing threshold and outputting image
for x in range(1, 5):
    sobelSumLoop = cv2.Sobel(gray, cv2.CV_64F, x, 0, ksize=5) + \
        cv2.Sobel(gray, cv2.CV_64F, 0, x, ksize=5)
    plt.subplot(1, 1, 1), plt.imshow(sobelSumLoop, cmap='gray')
    plt.title('Sobel Sum in For loop'), plt.xticks([]), plt.yticks([])
    plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
