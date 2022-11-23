#!/usr/bin/env python3

import cv2
import csv
import numpy as np
import pickle



### Get image directory from user
# img_dir = input("Please enter the image directory: ")
# f_type = input("Please enter the file type: ")

img_dir = '2022Simgs'
f_type = 'jpg'
### Load training images and labels

imageDirectory = './'+img_dir+'/'
file_type = '.'+f_type

def crop_img(img_path:str, thres_area:int, crop_size:tuple)->np.array:
    """Perform data preprocessing on image. Find all the contour in an image then crop based on the 
        convex contour that exceeds a given threshold area. Returns a black image if no good 
        contour is detected.

        Args:
            img_path  :   (str) representing the path of the input image
            thres_area:   (int) representing the minimum area for which to consider a contour as valid
            crop_size :   (int,int) representing the size of the desired cropped image
        Returns:
            crop_img  :   (inp_size,3) np.array representing the output image
            cnt_img   :   (img.shape) np.array representing the original image with contours
        """
    img = cv2.imread(img_path)
    
    # convert the image to a binary image and apply median filter to smoothe the edges.
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

    # Find contours, based on the edges detected on the binary image
    edges = cv2.Canny(blackAndWhiteImage, 50, 200)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    closed_contours = []
    enter=False
    for c in sorted_contours:
        x, y, w, h = cv2.boundingRect(c)
        cropped_contour = img[y:y + h, x:x + w]
        # print('contour area',cv2.contourArea(c))
        if cv2.contourArea(c)>=thres_area:
            closed_contours.append(c)
            if enter==False:
                enter=True
                crop_img=cropped_contour
        else:
            pass
    if len(closed_contours) ==0:
        crop_img = np.zeros(img.shape)
    # print('Image {}, Number of contours {}'.format(i,len(closed_contours)))

    # draw contour on the image
    cnt_img = img.copy()
    cv2.drawContours(image=cnt_img, contours=closed_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
   
    crop_img=cv2.resize(crop_img,crop_size,interpolation=cv2.INTER_LINEAR)
    return crop_img, cnt_img

try:
    with open(imageDirectory + 'train.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
except: 
    raise FileNotFoundError('Wrong directory or file type')


# An array with all the training images
train = np.zeros((len(lines),64,64,3))

for j in range(len(lines)):
    img = cv2.imread(imageDirectory +lines[j][0]+file_type)

    train[j,:,:,:],_ = crop_img(imageDirectory +lines[j][0]+file_type, thres_area=1500, crop_size=(64,64))


# here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
train_data = train.flatten().reshape(len(lines), 64*64*3)
# train_data = train.flatten().reshape(len(lines), 64*64)
train_data = train_data.astype(np.float32)

# read in training labels
train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])


# ### Train knn classifier
# knn = cv2.ml.KNearest_create()
# knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)


# Train the SVM open cv
cv_svm = cv2.ml.SVM_create()
cv_svm.setType(cv2.ml.SVM_C_SVC)
cv_svm.setC(0.2)
cv_svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setGamma(10)
cv_svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, int(1e4), 1e-6))
cv_svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Train the SVM sklean
# clf = svm.SVC(kernel='linear')
# clf.fit(train_data, train_labels)

# pickle.dump(cv_svm,open('model.pkl', 'wb'))
cv_svm.save('cv_svm')

if(__debug__):
	# Title_images = 'Original Image'
    Title_contour = 'Contour Image'
    Title_resized = 'Image Resized'
    cv2.namedWindow( Title_contour, cv2.WINDOW_AUTOSIZE )



### Run test images
try:
    with open(imageDirectory + 'test.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
except:
    raise FileNotFoundError('Wrong directory or file type')
  

correct = 0.0
confusion_matrix = np.zeros((6,6))

k = 9

for i in range(len(lines)):
    original_img = cv2.imread(imageDirectory+lines[i][0]+file_type)

    test_img, cnt_img =crop_img(imageDirectory +lines[i][0]+file_type, thres_area=1500, crop_size=(64,64))

    if(__debug__):
        cv2.imshow(Title_contour, cnt_img)
        cv2.imshow(Title_resized, test_img)
        key = cv2.waitKey()
        if key==27:    # Esc key to stop
            break
    test_img = test_img.flatten().reshape(1, 64*64*3)

    test_img = test_img.astype(np.float32)
    test_label = np.int32(lines[i][1])

    #knn prediction
    # ret, results, neighbours, dist = knn.findNearest(test_img, k)

    # open cv svm prediction
    _,ret = cv_svm.predict(test_img)
    if len(ret) == 0:
        ret = 0
    else:
        ret = ret[0][0]

    # sklearn svm prediction
    # dec = clf.decision_function(test_img)
    # ret = np.argmax(dec , axis = 1)[0]

    if test_label == ret:
        print(str(lines[i][0]) + " Correct, " + str(ret))
        correct += 1
        confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
    else:
        confusion_matrix[test_label][np.int32(ret)] += 1
        
        print(str(lines[i][0]) + " Wrong, " + str(test_label) + " classified as " + str(ret))
        # print("\tneighbours: " + str(neighbours))
        # print("\tdistances: " + str(dist))



print("\n\nTotal accuracy: " + str(correct/len(lines)))
print(confusion_matrix)
