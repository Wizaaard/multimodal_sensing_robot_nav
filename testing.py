import numpy as np
import cv2 as cv
import csv
import cv2
imageDirectory = './2022Fimgs/'

with open(imageDirectory + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)


for i in range(len(lines)):
    # img = cv.imread('66.png') # Read in your image
    img = cv.imread(imageDirectory +lines[i][0]+".png",0)

    enter=False
    # gray = cv.medianBlur(cv.cvtColor(img, cv.COLOR_RGB2GRAY), 5)
    # (thresh, blackAndWhiteImage) = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    (thresh, blackAndWhiteImage) = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
    edges = cv2.Canny(blackAndWhiteImage, 50, 200)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for (i, c) in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(c)
        cropped_contour = img[y:y + h, x:x + w]
        if enter==False:
            enter=True
            crop_img=cropped_contour

    train_image=cv2.resize(crop_img,(64, 64),interpolation=cv.INTER_LINEAR)


    # cv2.imwrite('tt_new.jpg', train_image)
    cv2.imshow('Image', train_image)
    # cv2.imshow('Image', train_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
