import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

nrows = 2
ncols = 2

#Cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
lowerBody_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')

#Face Dection
faceImg = cv2.imread('Friends.jpg')
color = cv2.cvtColor(faceImg, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

#People picture
bodyImg = cv2.imread('people.jpg')
color2 = cv2.cvtColor(bodyImg, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)

#Looking for Faces and eyes
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(color,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = color[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#Looking for body
lowerBody = lowerBody_cascade.detectMultiScale(color2)
for (x,y,w,h) in lowerBody:
            cv2.rectangle(color2,(x,y),(x+w,y+h),(255,0,0),2)

#Colour Picture
colImg = cv2.imread('city.jpg')
color3 = cv2.cvtColor(colImg, cv2.COLOR_BGR2RGB)

#Splitting picture
(B, G, R) = cv2.split(colImg)
zerosRGB = np.zeros(color3.shape[:2], dtype="uint8")

#Getting Only R G or B value
img3Red = cv2.merge([zerosRGB, zerosRGB, R])
img3Green = cv2.merge([zerosRGB, G, zerosRGB])
img3Blue = cv2.merge([B, zerosRGB, zerosRGB])

#Converting from RGB to HSV 
colImgHsv = cv2.cvtColor(color3, cv2.COLOR_RGB2HSV)

#Splitting HSV 
h,s,v = cv2.split(colImgHsv)

# #Plot - For Object Detection
plt.subplot(nrows, ncols,1),plt.imshow(color, cmap = 'gray')
plt.title('Face and Eye Detection'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,2),plt.imshow(color2, cmap = 'gray')
plt.title('LowerBody Detection'), plt.xticks([]), plt.yticks([])

plt.show()

#Plot for Color RGB split
plt.subplot(nrows, ncols,1),plt.imshow(img3Red, cmap = 'gray')
plt.title('Red'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,2),plt.imshow(img3Green, cmap = 'gray')
plt.title('Green'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,3),plt.imshow(img3Blue, cmap = 'gray')
plt.title('Blue'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,4),plt.imshow(color3, cmap = 'gray')
plt.title('Original RGB'), plt.xticks([]), plt.yticks([])

plt.show()

#Plot for Color HSV split
plt.subplot(nrows, ncols,1),plt.imshow(h, cmap = 'gray')
plt.title('Hue'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,2),plt.imshow(s, cmap = 'gray')
plt.title('Saturation'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,3),plt.imshow(v, cmap = 'gray')
plt.title('Value'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,4),plt.imshow(colImgHsv, cmap = 'gray')
plt.title('Original HSV'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()