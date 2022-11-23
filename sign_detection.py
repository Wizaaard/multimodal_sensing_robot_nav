#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


import numpy as np
import cv2
from cv_bridge import CvBridge

from geometry_msgs.msg import Point

class SignDetection(Node):

    def __init__(self):		
        # Creates the node.
        super().__init__('Sign Detection Subscriber')
        
        self.pos=Point()
        self.cnt_img = None

        self.cv_svm = cv2.ml.SVM_create().load('cv_svm')
        
        #Declare that the Sign Detection Subscriber node is subcribing to the /camera/image/compressed topic.
        self._video_subscriber = self.create_subscription(
                CompressedImage,
                '/camera/image/compressed',
                self._image_callback,
                1)
        self._video_subscriber # Prevents unused variable warning.
        
        self._point_publisher = self.create_publisher(
                Point,
                'img_point',
                10)
        print("publish")
        self._point_publisher # Prevents unused variable warning.
    
    def crop_img(self,img, thres_area:int, crop_size:tuple)->np.array:
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
        # img = cv2.imread(img_path)

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

    def _image_callback(self, CompressedImage):	
        print("entering callback")
        # Getting Robot image 
        self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
        frame=self._imgBGR

        # Preprocessing the image from the robot
        crop_frame, self.cnt_img = self.crop_img(frame,thres_area=1500, crop_size=(64,64))

        # Flatten the cropped image into a (1,N) vector
        crop_frame = crop_frame.flatten().reshape(1, 64*64*3)
        crop_frame = crop_frame.astype(np.float32)

        # Run the image through the classifier
        _,ret = self.cv_svm.predict(crop_frame)
        if len(ret) == 0:
            ret = 0
        else:
            ret = ret[0][0]
        
        self.pos.x=ret
        print("Label",self.pos.x)
        self._point_publisher.publish(self.pos)
        # self.get_logger().info('Publishing')

def main():
	rclpy.init() #init routine needed for ROS2.
	sign_subscriber = SignDetection() #Create class object to be used.
	
	rclpy.spin(sign_subscriber) # Trigger callback processing.		

	#Clean up and shutdown.
	sign_subscriber.destroy_node()  
	rclpy.shutdown()


if __name__ == '__main__':
	main()