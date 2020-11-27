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
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera2/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    # ACTUAL VALUE MOVEMENTS FOR SECTION 4.1:
    self.joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command",Float64,queue_size=10)

    # ESTIMATED VALUE MOVEMENTS FOR SECTION 4.1:
    self.joint1_estimated_pub = rospy.Publisher("/robot/joint1_estimated_position",Float64,queue_size=10)


  def detect_yellow(self, image1, image2):
    # Takes YZ plane
    yellow_mask_image1 = cv2.inRange(image1, (0,100,100), (35,255,255))
    M_image1 = cv2.moments(yellow_mask_image1)
    cy = int(M_image1['m10']/M_image1['m00'])
    cz = int(M_image1['m01']/M_image1['m00'])

    # Takes XZ plane
    yellow_mask_image2 = cv2.inRange(image2, (0,100,100), (35,255,255))
    M_image2 = cv2.moments(yellow_mask_image2)
    cx = int(M_image2['m10']/M_image2['m00'])
    cz = int(M_image2['m01']/M_image2['m00'])

    #im1 = cv2.imshow('image2_green',yellow_mask_image2)
    #cv2.circle(image1,(cy,cz), 2, (255,255,255), -1)
    #cv2.circle(image2, (cx,cz), 2, (255,255,255), -1)

    return np.array([cx,cy,cz])

  def detect_blue(self, image1, image2):
    # Takes YZ plane
    blue_mask_image1 = cv2.inRange(image1, (100,0,0), (255,50,50))
    M_image1 = cv2.moments(blue_mask_image1)
    cy = int(M_image1['m10']/M_image1['m00'])
    cz = int(M_image1['m01']/M_image1['m00'])
    
    # Takes XZ plane
    blue_mask_image2 = cv2.inRange(image2, (100,0,0), (255,50,50))
    M_image2 = cv2.moments(blue_mask_image2)
    cx = int(M_image2['m10']/M_image2['m00'])
    cz = int(M_image2['m01']/M_image2['m00'])

    return np.array([cx,cy,cz])

  def calculate_joint_angle_1(self, image1, image2):

    yellowJoint = self.detect_yellow(image1,image2)
    blueJoint = self.detect_blue(image1,image2)

    joint1 = yellowJoint - blueJoint
    # WARNING: Chamfer matching might be required...
    print("Joint 1 Changes")
    print(joint1)

    # Find vector perpendicular to rotation axis - Z axis
    normToZAxis = [-1,0,0]

    # Calculate dot product, obtain normalized vectors between the joints
    dotProduct = joint1[0] * normToZAxis[0] + joint1[1] * normToZAxis[1] + joint1[2] * normToZAxis[2]

    normalizedVector1 = np.sqrt(joint1[0]**2 + joint1[1]**2 + joint1[2]**2)
    normalizedVector2 = np.sqrt(normToZAxis[0]**2 + normToZAxis[1]**2 + normToZAxis[2]**2)

    x_angle = np.arccos(dotProduct / (normalizedVector1 * normalizedVector2))

    theta = np.arctan2(np.sin(x_angle), np.cos(x_angle))

    return theta


  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Attempt to publish results
    try:
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)

  # Recieve data, process it, and publish
  def callback2(self,data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    #im2=cv2.imshow('XZ plane- camera 2', self.cv_image2)
    #cv2.waitKey(1)

    ## Extra code can be inserted here
    #self.detect_yellow(self.cv_image1, self.cv_image2)

    ## SECTION 4.1:

    # ACTUAL VALUES
    joint1Value = Float64()
    joint1Value.data = ((np.pi) * np.sin((np.pi/15) * rospy.get_time()))
    self.joint1_pub.publish(joint1Value)

    # ESTIMATED VALUES
    joint1EstimatedValue = Float64()
    joint1EstimatedValue.data = self.calculate_joint_angle_1(self.cv_image1,self.cv_image2)
    self.joint1_estimated_pub.publish(joint1EstimatedValue)

    # Differences- used for debugging
    print(abs(joint1EstimatedValue.data - joint1Value.data))

    # Display images
    im1=cv2.imshow('window1', self.cv_image1)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
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


