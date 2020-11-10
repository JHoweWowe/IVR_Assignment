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

    # NEW: Initalize a publisher to send joints' angular position to a topic called joints_pos
    self.my_joints_pub = rospy.Publisher("my_joints_pos",Float64MultiArray, queue_size=10)
    
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    # Move (publish) the following joints with sinusoidal signals, as requested in coursework document
    self.joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command",Float64,queue_size=10)
    self.joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command",Float64,queue_size=10)
    self.joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command",Float64,queue_size=10)
  
    # Isolate the 4 blobs to help obtain 3 joints using color detection
    # TODO: #1- Make the blob detection algorithm more accurate- lessen iterations for dilation or remove dilation
    # TODO: #2- Momements may not be correctly shown (what happens if int(M['m00'] == 0??)- which affect the joint angle detection

  def detect_red(self, image):
    # Isolate red color- threshold slightly differs!!!
    red_mask = cv2.inRange(image, (0,0,100), (35,35,255))
    # Kernel convolution created to dilate the red_mask image for safety
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations = 1)

    # Obtain moments of binary image
    M = cv2.moments(red_mask)

    # Calculate pixel coordinates for blob center
    if (M['m00'] != 0):
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
    else:
      cx = self.detect_green(image)[0]
      cy = self.detect_green(image)[1]

    # Test function to show centroid
    #cv2.circle(image, (cx,cy), 3, (255,255,255), -1)
    #cv2.imshow('momentsWindow', image)
    #cv2.waitKey(1)

    # Test function to show image window of mask
    #im1 = cv2.imshow('red_mask_window', red_mask)
    #cv2.waitKey(1)

    return np.array([cx,cy])

  def detect_blue(self, image):
    # Isolate blue color
    blue_mask = cv2.inRange(image, (100,0,0), (255,50,50))
    # Kernel convolution created to dilate the blue_mask image
    kernel = np.ones((5,5), np.uint8)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations = 1)

    # Obtain moments of binary image
    M = cv2.moments(blue_mask)

    # Calculate pixel coordinates for blob center
    if (M['m00'] != 0):
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
    else:
      cx = self.detect_yellow(image)[0]
      cy = self.detect_yellow(image)[1]

    # Test function to show centroid
    #cv2.circle(image, (cx,cy), 3, (255,255,255), -1)
    #cv2.imshow('momentsWindow', image)
    #cv2.waitKey(1)

    # Test function to show image window of mask
    #im1 = cv2.imshow('blue_mask_window', blue_mask)
    #cv2.waitKey(1)

    return np.array([cx,cy])
  
  def detect_green(self, image):
    # Isolate green color- threshold slightly differs!
    green_mask = cv2.inRange(image, (0,100,0), (25,255,25))
    # Kernel convolution created to dilate the green_mask image
    kernel = np.ones((5,5), np.uint8)
    green_mask = cv2.dilate(green_mask, kernel, iterations = 1)

    # Obtain moments of binary image
    M = cv2.moments(green_mask)

    # Calculate pixel coordinates for blob center
    if (M['m00'] != 0):
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
    else:
      cx = self.detect_blue(image)[0]
      cy = self.detect_blue(image)[1]

    # Test function to show centroid
    #cv2.circle(image, (cx,cy), 3, (255,255,255), -1)
    #cv2.imshow('momentsWindow', image)
    #cv2.waitKey(1)

    # Test function to show image window of mask
    #im1 = cv2.imshow('green_mask_window', green_mask)
    #cv2.waitKey(1)

    return np.array([cx,cy])

  def detect_yellow(self, image):
    # Isolate yellow color- threshold slightly differs!
    yellow_mask = cv2.inRange(image, (0,100,100), (35,255,255))

    # Kernel convolution created to dilate the yellow_mask image
    kernel = np.ones((5,5), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations = 1)

    # Obtain moments of binary image
    M = cv2.moments(yellow_mask)

    # Calculate pixel coordinates for blob center
    if (M['m00'] != 0):
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
    else:
      cx = self.detect_red(image)[0]
      cy = self.detect_red(image)[1]

    #print(np.array([cx,cy]))

    # Test function to show centroid
    #cv2.circle(image, (cx,cy), 3, (255,255,255), -1)
    #cv2.imshow('momentsWindow', image)
    #cv2.waitKey(1)

    # Test function to show image window of mask
    #im1 = cv2.imshow('yellow_mask_window', yellow_mask)
    #cv2.waitKey(1)

    return np.array([cx,cy])

  # Section 2.2 of assignment:
  def detect_orange_target(self, image):
    # Obtain image for 2 moving objects - sphere and box
    orange_mask = cv2.inRange(image, (0,45,100), (15,90,150))
    
    # Apply Canny Edge Detection
    edged = cv2.Canny(orange_mask, 35, 195)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    for contour in contours:
      if (cv2.contourArea(contour) > maxArea):
        maxArea = cv2.contourArea(contour)
        sphere = contour

    M = cv2.moments(sphere)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # TODO: Find distance from sphere to the base (yellow sphere)

    return np.array([cx,cy])


  def pixel2MeterForLink1(self, image):
    joint1 = self.detect_yellow(image)
    joint2 = self.detect_blue(image)
    # Find squared distance (in pixels) between the blob joints
    dist = np.sum((joint1-joint2)**2)
    return 2.5 / np.sqrt(dist)

  # Should return 0 because length for link 2 is 0
  def pixel2MeterForLink2(self, image):
    return 0

  def pixel2MeterForLink3(self, image):
    joint2 = self.detect_blue(image)
    joint3 = self.detect_green(image)
    # Find squared distance (in pixels) between blob joints
    dist = np.sum((joint2-joint3)**2)
    return 3.5 / np.sqrt(dist)
  
  def pixel2MeterForLink4(self, image):
    joint3 = self.detect_green(image)
    joint4 = self.detect_red(image)
    # Find squared distance (in pixels) between blob joints
    dist = np.sum((joint3-joint4)**2)
    return 3 / np.sqrt(dist)

  def detect_joint_angle1(self, image):
    a = self.pixel2MeterForLink1(image)
    # Obtain center of each coloured blob in meters
    center = a * self.detect_yellow(image)
    circle1Pos = a * self.detect_blue(image)
    # Solve using trignometry
    ja1 = np.arctan2(center[0]-circle1Pos[0], center[1]-circle1Pos[1])
    #print("Joint Angle")
    #print(ja1)
    return ja1

  # Joint_angle 2 should ALWAYS be 0 NO???
  def detect_joint_angle2(self, image):
    return 0
  
  def detect_joint_angle3(self, image):
    a = self.pixel2MeterForLink3(image)
    # Obtain center of each coloured blob in meters
    circle1Pos = a * self.detect_blue(image)
    circle2Pos = a * self.detect_green(image)

    # Obtain other joint angles
    ja1 = self.detect_joint_angle1(image)
    ja2 = self.detect_joint_angle2(image)

    # Solve using trignometry
    ja3 = np.arctan2(circle1Pos[0] - circle2Pos[0], circle1Pos[1]-circle1Pos[1]) - ja2 - ja1
    return ja3

  def detect_joint_angle4(self, image):
    a = self.pixel2MeterForLink4(image)

    # Obtain center for each coloured blob
    circle2Pos = a * self.detect_green(image)
    circle3Pos = a * self.detect_red(image)

    # Obtain other joint angles
    ja1 = self.detect_joint_angle1(image)
    ja2 = self.detect_joint_angle2(image)
    ja3 = self.detect_joint_angle3(image)

    ja4 = np.arctan2(circle2Pos[0]-circle3Pos[0], circle2Pos[1]-circle3Pos[1]) - ja3 - ja2 - ja1
    return ja4

  def detect_all_joint_angles(self, image):
    ja1 = self.detect_joint_angle1(image)
    ja2 = self.detect_joint_angle2(image)
    ja3 = self.detect_joint_angle3(image)
    ja4 = self.detect_joint_angle4(image)
    return np.array([ja1,ja2,ja3,ja4])
  
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

    a = self.detect_all_joint_angles(self.cv_image1)

    # Test call image detection functions for showing masks respectively
    #self.detect_red(self.cv_image1)
    #self.detect_blue(self.cv_image1)
    #self.detect_green(self.cv_image1)
    #self.detect_yellow(self.cv_image1)
    self.detect_orange_target(self.cv_image1)

    # Obtain joints data when finished
    self.my_joints = Float64MultiArray()
    #self.joints.data = np.array([0,0,0])
    self.my_joints.data = a

    # ACTUAL VALUES
    # Set movement of joint values according to sinusoidal signals
    # and publish the movement values
    joint2Value = Float64()
    joint2Value.data = ((np.pi/2) * np.sin((np.pi/15) * rospy.get_time()))
    self.joint2_pub.publish(joint2Value)
    joint3Value = Float64()
    joint3Value.data = ((np.pi/2) * np.sin((np.pi/18) * rospy.get_time()))
    self.joint3_pub.publish(joint3Value)
    joint4Value = Float64()
    joint4Value.data = ((np.pi/2) * np.sin((np.pi/20) * rospy.get_time()))
    self.joint4_pub.publish(joint4Value)
    
    # MY VALUES
    
    # Differences between actual values and my values
    print("Differences:")
    print(abs(joint3Value.data - a[2]))
    print(abs(joint4Value.data - a[3]))


    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      self.my_joints_pub.publish(self.my_joints)
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


