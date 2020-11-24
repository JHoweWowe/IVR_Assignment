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
    
    # initialize a subscriber to recieve messages from a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    self.image_sub1 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    # ACTUAL movements: Publish actual joints states w.r.t sinusoidal signals
    self.joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command",Float64,queue_size=10)
    self.joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command",Float64,queue_size=10)
    self.joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command",Float64,queue_size=10)

    # ESTIMATED movements: Publish estimated joint states 
    self.joint2_estimate_pub = rospy.Publisher("/robot/joint2_estimated_position",Float64,queue_size=10)
    self.joint3_estimate_pub = rospy.Publisher("/robot/joint3_estimated_position",Float64,queue_size=10)
    self.joint4_estimate_pub = rospy.Publisher("/robot/joint4_estimated_position",Float64,queue_size=10)

    # SECTION 2.2: Publish target estimation
    self.target_x_estimate_pub = rospy.Publisher("/target/target_x_estimated_position",Float64,queue_size=10)
    self.target_y_estimate_pub = rospy.Publisher("/target/target_y_estimated_position",Float64,queue_size=10)
    self.target_z_estimate_pub = rospy.Publisher("/target/target_z_estimated_position",Float64,queue_size=10)
  
  def detect_red(self, image1, image2):
    red_mask_image1 = cv2.inRange(image1, (0,0,100), (35,35,255))
    M_image1 = cv2.moments(red_mask_image1)

    if (M_image1['m00'] != 0):
      cy = int(M_image1['m10']/M_image1['m00'])
      cz = int(M_image1['m01']/M_image1['m00'])
    else:
      cy = self.detect_green(image1,image2)[1]
      cz = self.detect_green(image1,image2)[2]

    red_mask_image2 = cv2.inRange(image2, (0,0,100), (35,35,255))
    M_image2 = cv2.moments(red_mask_image2)

    if (M_image2['m00'] != 0):
      cx = int(M_image2['m10']/M_image2['m00'])
      cz = int(M_image2['m01']/M_image2['m00'])
    else:
      cx = self.detect_green(image1,image2)[0]
      cz = self.detect_green(image1,image2)[2]

    #print("Dimensions for red blob:")
    #print(np.array([cx,cy,cz]))

    return np.array([cx,cy,cz])

  def detect_blue(self, image1, image2):
    # Isolate blue color
    blue_mask_image1 = cv2.inRange(image1, (100,0,0), (255,50,50))

    # Obtain moments of binary image
    M_image1 = cv2.moments(blue_mask_image1)

    # Calculate pixel coordinates for blob center
    if (M_image1['m00'] != 0):
      cy = int(M_image1['m10']/M_image1['m00'])
      cz = int(M_image1['m01']/M_image1['m00'])
    else:
      cy = self.detect_yellow(image1, image2)[1]
      cz = self.detect_yellow(image1, image2)[2]

    blue_mask_image2 = cv2.inRange(image2, (100,0,0), (255,50,50))
    M_image2 = cv2.moments(blue_mask_image2)

    # Calculate pixel coordinates for blob center
    if (M_image2['m00'] != 0):
      cx = int(M_image2['m10']/M_image2['m00'])
      cz = int(M_image2['m01']/M_image2['m00'])
    else:
      cx = self.detect_yellow(image1,image2)[0]
      cz = self.detect_yellow(image1,image2)[2]

    #print("Dimensions for blue blob:")
    #print(np.array([cx,cy,cz]))

    #cv2.circle(image1,(cy,cz), 3, (255,255,255), -1)
    #cv2.circle(image2, (cx,cz), 3, (255,255,255), -1)

    return np.array([cx,cy,cz])
  
  def detect_green(self, image1, image2):
    # Isolate green color- threshold slightly differs!
    green_mask_image1 = cv2.inRange(image1, (0,100,0), (40,255,40))

    # Obtain moments of binary image
    M_image1 = cv2.moments(green_mask_image1)
    # Calculate pixel coordinates for blob center
    if (M_image1['m00'] != 0):
      cy = int(M_image1['m10']/M_image1['m00'])
      cz = int(M_image1['m01']/M_image1['m00'])
    else:
      cy = self.detect_blue(image1,image2)[1]
      cz = self.detect_blue(image1,image2)[2]

    green_mask_image2 = cv2.inRange(image2, (0,100,0), (40,255,40))

    # Currently problem with detecting moments centroid at green blob detection at camera 2
    contours, hierarchy = cv2.findContours(green_mask_image2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    M_image2 = cv2.moments(contours[0])
    if (M_image2['m00'] != 0):
      cx = int(M_image2['m10']/M_image2['m00'])
      cz = int(M_image2['m01']/M_image2['m00'])
    else:
      cx = self.detect_blue(image1,image2)[0]
      cz = self.detect_blue(image1,image2)[2]

    #print("Dimensions for green blob:")
    #print(np.array([cx,cy,cz]))

    #im1 = cv2.imshow('image2_green',green_mask_image2)
    #cv2.circle(image1,(cy,cz), 2, (255,255,255), -1)
    #cv2.circle(image2, (cx,cz), 2, (255,255,255), -1)

    return np.array([cx,cy,cz])

  # Testing purposes only
  def detect_yellow(self, image1, image2):
    yellow_mask_image1 = cv2.inRange(image1, (0,100,100), (35,255,255))
    M_image1 = cv2.moments(yellow_mask_image1)
    cy = int(M_image1['m10']/M_image1['m00'])
    cz = int(M_image1['m01']/M_image1['m00'])

    yellow_mask_image2 = cv2.inRange(image2, (0,100,100), (35,255,255))
    M_image2 = cv2.moments(yellow_mask_image2)
    cx = int(M_image2['m10']/M_image2['m00'])

    return np.array([cx,cy,cz])

  # SECTION 2.2: Target detection
  def detect_orange_sphere(self, image1, image2):

    # Get orange mask from YZ plane- camera1
    orange_mask_image1 = cv2.inRange(image1, (0,45,100), (15,90,150))
    # Get orange mask from XZ plane- camera2
    orange_mask_image2 = cv2.inRange(image2, (0,45,100), (15,90,150))

    # Apply Canny Edge detection based on the image mask given as threshold
    edged_shapes_YZ = cv2.Canny(orange_mask_image1, 40, 180)
    edged_shapes_XZ = cv2.Canny(orange_mask_image2, 40, 180)
    
    contours_YZ, _ = cv2.findContours(edged_shapes_YZ.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_XZ, _ = cv2.findContours(edged_shapes_XZ.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #test = cv2.drawContours(image1, contours_YZ, 0, (0,255,0), 3)
    sphere_YZ = contours_YZ[0]
    sphere_XZ = contours_XZ[0]

    M_YZ = cv2.moments(sphere_YZ)

    # If moments cannot be detected properly, it would obtain centroid from previous estimates
    last_cx = 0
    last_cy = 0
    last_cz = 0

    if (M_YZ['m00'] != 0):
      cy = int(M_YZ['m10']/M_YZ['m00'])
      cz = int(M_YZ['m01']/M_YZ['m00'])
      last_cy = cy
      last_cz = cz
    else:
      cy = last_cy
      cz = last_cz

    M_XZ = cv2.moments(sphere_XZ)

    if (M_XZ['m00'] != 0):
      cx = int(M_XZ['m10']/M_XZ['m00'])
      cz = int(M_XZ['m01']/M_XZ['m00']) 
      last_cx = cx
      last_cz = cz
    else:
      cx = last_cx
      cz = last_cz

    return np.array([cx,cy,cz])

  # Rotates link 2 based on X axis- use camera 1
  def pixel2MeterForLink2(self, image1, image2):
    joint2 = self.detect_blue(image1, image2)
    joint3 = self.detect_green(image1, image2)
    # Finds Pythagorean distance between two vectors
    dist = np.linalg.norm(joint3 - joint2)
    return 3.5 / dist

  # Rotates link 3 based on Y axis- will need to implement camera 2
  def pixel2MeterForLink3(self, image1, image2):
    joint2 = self.detect_blue(image1, image2)
    joint3 = self.detect_green(image1, image2)
    # Finds Pythagorean distance between two vectors
    dist = np.linalg.norm(joint3 - joint2)
    return 3.5 / dist

  # Rotates link 4 based on X axis- will need to use camera 1
  def pixel2MeterForLink4(self, image1, image2):
    joint3 = self.detect_green(image1, image2)
    joint4 = self.detect_red(image1, image2)
    # Pythagorean distance between two vectors
    dist = np.linalg.norm(joint4 - joint3)
    return 3 / dist

  # Joint angle 2
  def detect_joint_angle2(self, image1, image2):

    # Detection problem with images- should be done with image1 ONLY
    blueBlob = self.detect_blue(image1, image2)
    greenBlob = self.detect_green(image1, image2)

    joint2 = greenBlob - blueBlob

    # Create an axis perpendicular to rotation axis- X
    normToXAxis = [0, 0, -1]

    dotProduct = normToXAxis[1] * joint2[1]  + normToXAxis[2] * joint2[2]

    normalizedVector1 = np.sqrt(normToXAxis[1]**2+normToXAxis[2]**2)
    normalizedVector2 = np.sqrt(joint2[1]**2+joint2[2]**2) 

    x_angle = np.arccos(dotProduct / (normalizedVector1 * normalizedVector2))

    theta1 = np.arctan2(np.sin(x_angle), np.cos(x_angle))
    if (joint2[1] < 0):
      theta1 = theta1
    else:
      theta1 = (-1) * theta1
    return theta1

  # Joint angle 3
  def detect_joint_angle3(self, image1, image2):

    blueBlob = self.detect_blue(image1,image2)
    greenBlob = self.detect_green(image1,image2)

    joint3 = greenBlob - blueBlob

    #Create axis perpendicular to rotation axis- Y
    normToYAxis = [0, 0, -1]

    dotProduct = normToYAxis[0] * joint3[0] + normToYAxis[2] * joint3[2]

    normalizedVector1 = np.sqrt(normToYAxis[0]**2+normToYAxis[2]**2)
    normalizedVector2 = np.sqrt(joint3[0]**2+joint3[2]**2)

    y_angle = np.arccos(dotProduct / (normalizedVector1 * normalizedVector2))

    beta = np.arctan2(np.sin(y_angle),np.cos(y_angle))
    if (joint3[0] >= 0):
      beta = beta
    else:
      beta = (-1) * beta
    
    return beta

  # Joint_angle 4
  def detect_joint_angle4(self, image1, image2):

    # Obtain the joint link between green blob and red blob
    blueBlob = self.detect_blue(image1,image2)
    greenBlob = self.detect_green(image1,image2)
    redBlob = self.detect_red(image1,image2)

    joint4 = redBlob - greenBlob

    sharedJoint = greenBlob - blueBlob

    # Projection from joint4 onto the shared joint
    dotProduct1 = joint4[0] * sharedJoint[0] + joint4[1] * sharedJoint[1] + joint4[2] * sharedJoint[2]
    normalizedVector1 = np.sqrt(sharedJoint[0]**2 + sharedJoint[1]**2 + sharedJoint[2]**2)

    vectorProjection = np.multiply((dotProduct1 / normalizedVector1), sharedJoint)

    #print(vectorProjection)

    dotProduct2 = vectorProjection[0] * joint4[0] + vectorProjection[1] * joint4[1] + vectorProjection[2] * joint4[2]
    newNormalizedVector1 = np.sqrt(joint4[0]**2+joint4[1]**2+joint4[2]**2)
    newNormalizedVector2 = np.sqrt(vectorProjection[0]**2+vectorProjection[1]**2+vectorProjection[2]**2)
    
    angle = np.arccos(dotProduct2 / (newNormalizedVector1 * newNormalizedVector2))

    # Currently angle doesn't take into consideration for some quadrants...but recording is good enough
    theta1 = np.arctan2(np.sin(angle), np.cos(angle))
    if (joint4[1] < 0):
      theta1 = theta1
    else:
      theta1 = (-1) * theta1
    return theta1

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))

    except CvBridgeError as e:
      print(e)
  
  # Recieve data from camera 2, process it, and publish
  def callback2(self, data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # ACTUAL VALUES
    # Set movement of joint values according to sinusoidal signals and publish the movement values
    # NOTE: All joints work - record for minimum 5 seconds only, more than 15 will be off
    
    # joint2Value = Float64()
    # joint2Value.data = ((np.pi/2) * np.sin((np.pi/15) * rospy.get_time()))
    # #joint2Value.data = (np.pi/5)
    # self.joint2_pub.publish(joint2Value)

    # joint3Value = Float64()
    # joint3Value.data = ((np.pi/2) * np.sin((np.pi/18) * rospy.get_time()))
    # #joint3Value.data = (np.pi/3)
    # self.joint3_pub.publish(joint3Value)

    # joint4Value = Float64()
    # joint4Value.data = ((np.pi/2) * np.sin((np.pi/20) * rospy.get_time()))
    # self.joint4_pub.publish(joint4Value)

    ## SECTION 2.1:
      
    # MY ESTIMATED VALUES
    # joint2EstimatedValue = Float64()
    # joint2EstimatedValue.data = self.detect_joint_angle2(self.cv_image1, self.cv_image2)
    # self.joint2_estimate_pub.publish(joint2EstimatedValue)

    # joint3EstimatedValue = Float64()
    # joint3EstimatedValue.data = self.detect_joint_angle3(self.cv_image1, self.cv_image2)
    # self.joint3_estimate_pub.publish(joint3EstimatedValue)

    # joint4EstimatedValue = Float64()
    # joint4EstimatedValue.data = self.detect_joint_angle4(self.cv_image1, self.cv_image2)
    # self.joint4_estimate_pub.publish(joint4EstimatedValue)

    # DIFFERENCES BETWEEN ACTUAL VALUES AND ESTIMATED VALUES
    #print("Differences between Joint2 actual and estimated joint angle values:")
    #print(abs(joint2Value.data - joint2EstimatedValue.data))

    # print("Differences between Joint3 actual and estimated joint angle values:")
    # print(abs(joint3Value.data - joint3EstimatedValue.data))

    #print("Differences between Joint4 actual and estimated joint angle values:")
    #print(abs(joint4Value.data - joint4EstimatedValue.data))

    ## SECTION 2.2:
    targetXEstimatedValue = Float64()
    targetXEstimatedValue.data = self.detect_orange_sphere(self.cv_image1, self.cv_image2)[0]
    self.target_x_estimate_pub.publish(targetXEstimatedValue)
    targetYEstimatedValue = Float64()
    targetYEstimatedValue.data = self.detect_orange_sphere(self.cv_image1, self.cv_image2)[1]
    self.target_y_estimate_pub.publish(targetYEstimatedValue)
    targetZEstimatedValue = Float64()
    targetZEstimatedValue.data = self.detect_orange_sphere(self.cv_image1, self.cv_image2)[2]
    self.target_z_estimate_pub.publish(targetZEstimatedValue)

    # Display images
    im1=cv2.imshow('window1', self.cv_image1)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    # Publish the results
    # try: 
    #   self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
    # except CvBridgeError as e:
    #   print(e)

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


