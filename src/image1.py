#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
  
    # Isolate the 4 blobs to help obtain 3 joints using color detection
  def detect_red(self, image):
    # Isolate red color- threshold slightly differs!!!
    red_mask = cv2.inRange(image, (0,0,100), (35,35,255))
    # Kernel convolution created to dilate the red_mask image for safety
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations = 1)

    # Obtain moments of binary image
    M = cv2.moments(red_mask)

    # Calculate pixel coordinates for blob center
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # Test function to show image window of mask
    im1 = cv2.imshow('red_mask_window', red_mask)
    cv2.waitKey(1)

    return np.array([cx,cy])

  def detect_blue(self, image):
    # Isolate blue color
    blue_mask = cv2.inRange(image, (100,0,0), (255,75,75))
    # Kernel convolution created to dilate the blue_mask image
    kernel = np.ones((5,5), np.uint8)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations = 3)

    # Obtain moments of binary image
    M = cv2.moments(blue_mask)

    # Calculate pixel coordinates for blob center
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # Test function to show image window of mask
    im1 = cv2.imshow('blue_mask_window', blue_mask)
    cv2.waitKey(1)

    return np.array([cx,cy])
  
  def detect_green(self, image):
    # Isolate green color- threshold slightly differs!
    green_mask = cv2.inRange(image, (0,100,0), (25,255,25))
    # Kernel convolution created to dilate the green_mask image
    kernel = np.ones((5,5), np.uint8)
    green_mask = cv2.dilate(green_mask, kernel, iterations = 3)

    # Obtain moments of binary image
    M = cv2.moments(green_mask)

    # Calculate pixel coordinates for blob center
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # Test function to show image window of mask
    im1 = cv2.imshow('green_mask_window', green_mask)
    cv2.waitKey(1)

    return np.array([cx,cy])

  def detect_yellow(self, image):
    # Isolate yellow color- threshold slightly differs!
    yellow_mask = cv2.inRange(image, (0,100,100), (0,255,255))

    # Kernel convolution created to dilate the yellow_mask image
    kernel = np.ones((5,5), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations = 3)

    # Obtain moments of binary image
    M = cv2.moments(yellow_mask)

    # Calculate pixel coordinates for blob center
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # Test function to show image window of mask
    im1 = cv2.imshow('yellow_mask_window', yellow_mask)
    cv2.waitKey(1)

    return np.array([cx,cy])


  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    im1=cv2.imshow('window1', self.cv_image1)
    cv2.waitKey(1)

    # Test call image detection functions for showing masks respectively
    self.detect_red(self.cv_image1)
    self.detect_blue(self.cv_image1)
    self.detect_green(self.cv_image1)
    self.detect_yellow(self.cv_image1)

    # TODO: Convert image to HSV format for clearer object detection and then publish it for joints

    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


