#!/usr/bin/env python3

import cv2
import sys
import csv
import time
import numpy as np

### Load training images and labels

# imageDirectory = './2022Fimgs/'
imageDirectory = './2022Simgs/'

with open(imageDirectory + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

# this line reads in all images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels
# train = np.array([np.array(cv2.resize(cv2.imread(imageDirectory +lines[i][0]+".png",0),(33,25))) for i in range(len(lines))])
# train = np.zeros((len(lines),64,64,3))
train = np.zeros((len(lines),64,64))

for j in range(len(lines)):
    img = cv2.imread(imageDirectory +lines[j][0]+".jpg",0)

    enter=False
    # gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
    # (thresh, blackAndWhiteImage) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    (thresh, blackAndWhiteImage) = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(blackAndWhiteImage, 50, 200)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for (i, c) in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(c)
        cropped_contour = img[y:y + h, x:x + w]
        if enter==False:
            enter=True
            crop_img=cropped_contour
    # print('trainig',j)
    train_image=cv2.resize(crop_img,(64,64),interpolation=cv2.INTER_LINEAR)
    # train[j,:,:,:] = train_image
    train[j,:,:] = train_image


# here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
# train_data = train.flatten().reshape(len(lines), 64*64*3)
train_data = train.flatten().reshape(len(lines), 64*64)
train_data = train_data.astype(np.float32)

# read in training labels
train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])


### Train classifier
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

if(__debug__):
	Title_images = 'Original Image'
	Title_resized = 'Image Resized'
	cv2.namedWindow( Title_images, cv2.WINDOW_AUTOSIZE )


# imageDirectory = './2022Fimgs/'
imageDirectory = './2022Simgs/'

### Run test images
with open(imageDirectory + 'test.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

correct = 0.0
confusion_matrix = np.zeros((6,6))

k = 5

for i in range(len(lines)):
    original_img = cv2.imread(imageDirectory+lines[i][0]+".jpg",0)
    # img = cv2.imread(imageDirectory +lines[j][0]+".png")
    # test_img = np.array(cv2.resize(cv2.imread(imageDirectory+lines[i][0]+".png",0),(33,25)))

    # img = cv2.imread(imageDirectory +lines[93][0]+".png")

    enter=False
    # gray = cv2.medianBlur(cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY), 5)
    # (thresh, blackAndWhiteImage) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    (thresh, blackAndWhiteImage) = cv2.threshold(original_img, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(blackAndWhiteImage, 50, 200)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for (j, c) in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(c)
        cropped_contour = original_img[y:y + h, x:x + w]
        if enter==False:
            enter=True
            crop_img=cropped_contour

    test_img=cv2.resize(crop_img,(64,64),interpolation=cv2.INTER_LINEAR)

    if(__debug__):
        cv2.imshow(Title_images, original_img)
        cv2.imshow(Title_resized, test_img)
        key = cv2.waitKey()
        if key==27:    # Esc key to stop
            break
    # test_img = test_img.flatten().reshape(1, 64*64*3)
    test_img = test_img.flatten().reshape(1, 64*64)
    test_img = test_img.astype(np.float32)

    test_label = np.int32(lines[i][1])

    ret, results, neighbours, dist = knn.findNearest(test_img, k)

    if test_label == ret:
        print(str(lines[i][0]) + " Correct, " + str(ret))
        correct += 1
        confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
    else:
        confusion_matrix[test_label][np.int32(ret)] += 1
        
        print(str(lines[i][0]) + " Wrong, " + str(test_label) + " classified as " + str(ret))
        print("\tneighbours: " + str(neighbours))
        print("\tdistances: " + str(dist))



print("\n\nTotal accuracy: " + str(correct/len(lines)))
print(confusion_matrix)
