#!/usr/bin/env python3

import cv2
import sys
import csv
import time
import numpy as np

### Load training images and labels
sift = cv2.ORB_create()
def fd_sift(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kps, des = sift.detectAndCompute(image, None)
    return des if des is not None else np.array([]).reshape(0, 128)


imageDirectory = './2022Fimgs/'

with open(imageDirectory + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

global_train_features = []
# fixed_size = (33,25,3)
for i in range(len(lines)):
    image = cv2.imread(imageDirectory +lines[i][0]+".png")
    # image.resize(fixed_size)
    fv_sift = fd_sift(image)
    global_feature = np.hstack([fv_sift])
    # global_feature.resize(fixed_size)
    global_train_features.append(global_feature)


# this line reads in all images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels
# train = []
# for i in range(len(lines)):

#     orig_img = cv2.imread(imageDirectory +lines[i][0]+".png")
#     w,h,_ = orig_img.shape
#     crop_w = int(w*0.3)
#     crop_h = int(h*0.3)
#     x = w//2
#     y = h//2

#     crop_img = orig_img[x-crop_w:x+crop_w, y-crop_h:y+crop_h]
#     train.append(cv2.resize(crop_img,(33,25)))

# train = np.array([np.array(cv2.resize(img,(33,25))) for i in range(len(lines))])
train = np.asarray(global_train_features)
print('train shape',train.shape)
print('train type',type(train))

# here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
train_data = train.flatten().reshape(len(lines), -1)
# train_data = train_data.astype(np.float32)

# read in training labels
train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])


### Train classifier
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

if(__debug__):
	Title_images = 'Original Image'
	Title_resized = 'Image Resized'
	cv2.namedWindow( Title_images, cv2.WINDOW_AUTOSIZE )


imageDirectory = './2022Fimgs/'

### Run test images
with open(imageDirectory + 'test.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

correct = 0.0
confusion_matrix = np.zeros((6,6))

k = 21

for i in range(len(lines)):
    # original_img = cv2.imread(imageDirectory+lines[i][0]+".png",0)
    # test_img = np.array(cv2.resize(cv2.imread(imageDirectory+lines[i][0]+".png",0),(33,25)))
   
    # test_img = []
    
    # original_img = cv2.imread(imageDirectory +lines[i][0]+".png")
    # h,w,_ = orig_img.shape
    # crop_w = int(w*0.3)
    # crop_h = int(h*0.3)
    # x = h//2
    # y = w//2

    # crop_img = original_img[x-crop_h:x+crop_h, y-crop_w:y+crop_w]
    # test_img.append(cv2.resize(crop_img,(33,25)))
    # test_img = np.asarray(test_img)
    # test_img = test_img[0,:,:,:]

    global_test_features = []
    original_img = cv2.imread(imageDirectory +lines[i][0]+".png")
    # original_img.resize(fixed_size)
    fv_sift_test = fd_sift(original_img)
    global_test_feature = np.hstack([fv_sift_test])
    # global_test_feature.resize(fixed_size)
    global_test_features.append(global_test_feature)

    # print('test shape',test_img.shape)
    # print('test type',type(test_img))

    test_img = np.asarray(global_test_features)
    test_img = test_img[0,:,:,:]

    if(__debug__):
        cv2.imshow(Title_images, original_img)
        # cv2.imshow(Title_resized, test_img)
        key = cv2.waitKey()
        if key==27:    # Esc key to stop
            break
    test_img = test_img.flatten().reshape(1, -1)
    # test_img = test_img.astype(np.float32)

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
