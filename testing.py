import numpy as np
import cv2 as cv
import csv
import cv2
imageDirectory = './2022Fimgs/'

with open(imageDirectory + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)


img = cv.imread('97.png') # Read in your image
# ksize = (10, 10)
# Using cv2.blur() method
# img = cv.blur(img, ksize)
gray = cv.medianBlur(cv.cvtColor(img, cv.COLOR_RGB2GRAY), 5)
(thresh, blackAndWhiteImage) = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
edges = cv2.Canny(blackAndWhiteImage, 50, 200)
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
for (i, c) in enumerate(sorted_contours):
    x, y, w, h = cv2.boundingRect(c)
    cropped_contour = img[y:y + h, x:x + w]

    cv2.imshow('Image', cropped_contour)
    cv2.waitKey(0)

cv2.destroyAllWindows()

#
# cv.imshow('Output', blackAndWhiteImage)
# cv.waitKey(0)
# cv.destroyAllWindows()














# _, mask = cv2.threshold(gray, thresh=150, maxval=255, type=cv2.THRESH_BINARY)
# im_thresh_gray = cv2.bitwise_and(gray, mask)
# # threshold to get just the signature
#
#
# # find where the signature is and make a cropped region
# points = np.argwhere(im_thresh_gray==0) # find where the black pixels are
# points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
# x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
# crop = img[y:y+h, x:x+w]
# cv2.imshow('save.jpg', im_thresh_gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
# # # img = cv2.imread(imageDirectory + lines[0][0] + ".png")
# # # contours, _ = cv2.findContours(...) # Your call to find the contours using OpenCV 2.4.x
# # _, contours, _ = cv2.findContours(img) # Your call to find the contours
# # idx = ... # The index of the contour that surrounds your object
# # mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
# # cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
# # out = np.zeros_like(img) # Extract out the object and place into output image
# # out[mask == 255] = img[mask == 255]
# #
# # # Now crop
# # (y, x) = np.where(mask == 255)
# # (topy, topx) = (np.min(y), np.min(x))
# # (bottomy, bottomx) = (np.max(y), np.max(x))
# # out = out[topy:bottomy+1, topx:bottomx+1]
#
# # Show the output image
# # cv2.imshow('Output', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()