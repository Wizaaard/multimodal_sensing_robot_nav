import numpy as np
import cv2 
import csv
# imageDirectory = './2022Fimgs/'
imageDirectory = './2022Simgs/'

with open(imageDirectory + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)


for i in range(len(lines)):
# for i in [52,68,69,72,73,99,128,174,175,176,177,178]:

    # img = cv.imread('66.png') # Read in your image
    img = cv2.imread(imageDirectory +lines[i][0]+".jpg")

    enter=False
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    # (thresh, blackAndWhiteImage) = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(blackAndWhiteImage, 50, 200)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    closed_contours = []
    for c in sorted_contours:
        x, y, w, h = cv2.boundingRect(c)
        cropped_contour = img[y:y + h, x:x + w]
        # print('contour area',cv2.contourArea(c))
        if cv2.contourArea(c)>=1500:
            closed_contours.append(c)
            if enter==False:
                enter=True
                crop_img=cropped_contour
        else:
            pass
    if len(closed_contours) ==0:
        crop_img = np.zeros(img.shape)
    # print('Image {}, Number of contours {}'.format(i,len(closed_contours)))

    image_copy = img.copy()
    cv2.drawContours(image=image_copy, contours=closed_contours
    
    , contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
   
    train_image=cv2.resize(crop_img,(64, 64),interpolation=cv2.INTER_LINEAR)

    if(__debug__):
        cv2.imshow('Original Image', blackAndWhiteImage)
        cv2.imshow('Cropped Image', train_image)
        cv2.imshow('Contour Image', image_copy)
        key = cv2.waitKey()
        if key==27:    # Esc key to stop
            break
    # cv2.imwrite('tt_new.jpg', train_image)
    # cv2.imshow('Image', train_image)
    # # cv2.imshow('Image', train_image)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
