#Lab9

# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import argparse
import random as rng

# Variables
nrows = 3
ncols = 3

# Import Image
img = cv2.imread('ATU1.jpg')

# Create Deep Copy
harrisImg = copy.deepcopy(img)
shiTImg = copy.deepcopy(img)

# GreyScale Image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris Corner Detection
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Plot Corners
threshold = 0.01  # number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(harrisImg, (j, i), 3, (0, 0, 255), -1)

# Shi Tomasi Corner Detection
corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
# Plot Corners
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv2.circle(shiTImg, (x, y), 3, (0, 0, 255), -1)

# ORB 
orb = cv2.ORB_create()
kp = orb.detect(img,None) # find the keypoints with ORB
kp, des = orb.compute(img, kp) # compute the descriptors with ORB
# draw only keypoints location,not size and orientation
orbImg = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

#BruteForceMatcher
atu1 = cv2.imread('atu1.jpg',cv2.IMREAD_GRAYSCALE) 
atu2 = cv2.imread('atu2.jpg',cv2.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(atu1,None)
kp2, des2 = orb.detectAndCompute(atu2,None)

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

# Sort in order of distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw 
atuMATCHED = cv2.drawMatches(atu1,kp1,atu2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#Own Images BruteForce
own1 = cv2.imread('Farm.jpg',cv2.IMREAD_GRAYSCALE) 
own2 = cv2.imread('Cliff.jpg',cv2.IMREAD_GRAYSCALE)

orb2 = cv2.ORB_create()
keyP1, d1 = orb.detectAndCompute(own1,None)
keyP2, d2 = orb.detectAndCompute(own2,None)
bf2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches2 = bf2.match(d1,d2)
matches2 = sorted(matches2, key = lambda x:x.distance)
ownMATCHED = cv2.drawMatches(own1,keyP1,own2,keyP2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#FLANN ATU
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(atu1,None)
kp2, des2 = sift.detectAndCompute(atu2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                matchesMask = matchesMask,
                flags = cv2.DrawMatchesFlags_DEFAULT)
atuFlannMatched = cv2.drawMatchesKnn(atu1,kp1,atu2,kp2,matches,None,**draw_params)

#Own Flann
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(own1,None)
kp2, des2 = sift.detectAndCompute(own2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                matchesMask = matchesMask,
                flags = cv2.DrawMatchesFlags_DEFAULT)
ownFlannMatched = cv2.drawMatchesKnn(own1,kp1,own2,kp2,matches,None,**draw_params)

rng.seed(12345)
#Contour Detection
def thresh_callback(val):
        threshold = val 

        # Detect edges using Canny
        canny_output = cv2.Canny(own1, threshold, threshold * 2)

        # Find contours
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
                color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        
        # Show in a window
        cv2.imshow('Contours', drawing)

# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help='Path to input image.', default='Farm.jpg')
args = parser.parse_args()

src = cv2.imread(cv2.samples.findFile(args.input))

if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)

# Convert image to gray and blur it
gray2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray2 = cv2.blur(gray2, (3,3))

# Plot Multiple Images
plt.subplot(nrows, ncols, 1), plt.imshow(
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 2), plt.imshow(gray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 3), plt.imshow(
    cv2.cvtColor(harrisImg, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Harris Corner'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 4), plt.imshow(
    cv2.cvtColor(shiTImg, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Shi Tomasi Corner'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 5), plt.imshow(
    cv2.cvtColor(orbImg, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('ORB'), plt.xticks([]), plt.yticks([])

plt.show()

# Plot Brute Force
plt.subplot(2, 1, 1), plt.imshow(atuMATCHED)
plt.title('BruteForceMatcher ATU'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 1, 2), plt.imshow(ownMATCHED)
plt.title('BruteForceMatcher Own Images'), plt.xticks([]), plt.yticks([])

plt.show()

#Plot FLANN
plt.subplot(2, 1, 1), plt.imshow(atuFlannMatched)
plt.title('FLANN ATU'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 1, 2), plt.imshow(ownFlannMatched)
plt.title('FLANN Own Images'), plt.xticks([]), plt.yticks([])

plt.show()

# Create Window for Colour Detection
source_window = 'Source'
cv2.namedWindow(source_window)
cv2.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv2.waitKey()