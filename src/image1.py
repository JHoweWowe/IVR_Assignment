#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import sympy as sp
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
from rospy.numpy_msg import numpy_msg


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)

    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    
    # initialize a subscriber to recieve messages from a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    # ACTUAL movements: Publish actual joints states w.r.t sinusoidal signals
    self.joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command",Float64,queue_size=10)
    self.joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command",Float64,queue_size=10)
    self.joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command",Float64,queue_size=10)
    self.joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command",Float64,queue_size=10)

    self.joints_ac = rospy.Subscriber("/robot/joint_states",JointState,self.updateAngles)
    self.target_x = rospy.Subscriber("target/x_position_controller/command",Float64,self.targetx)
    self.target_y = rospy.Subscriber("target/y_position_controller/command",Float64,self.targety)
    self.target_z = rospy.Subscriber("target/z_position_controller/command",Float64,self.targetz)



    # ESTIMATED movements: Publish estimated joint states 
    self.joint2_estimate_pub = rospy.Publisher("/robot/joint2_estimated_position",Float64,queue_size=10)
    self.joint3_estimate_pub = rospy.Publisher("/robot/joint3_estimated_position",Float64,queue_size=10)
    self.joint4_estimate_pub = rospy.Publisher("/robot/joint4_estimated_position",Float64,queue_size=10)

    # SECTION 2.2: Publish target estimation
    self.target_x_estimate_pub = rospy.Publisher("/target/target_x_estimated_position",Float64,queue_size=10)
    self.target_y_estimate_pub = rospy.Publisher("/target/target_y_estimated_position",Float64,queue_size=10)
    self.target_z_estimate_pub = rospy.Publisher("/target/target_z_estimated_position",Float64,queue_size=10)


    # SECTION 3.2: Publish end effector calculated position using inverse kinematics
    self.end_effector_x_pub = rospy.Publisher("/robot/end_effector_x_estimated",Float64,queue_size=10)
    self.end_effector_y_pub = rospy.Publisher("/robot/end_effector_y_estimated",Float64,queue_size=10)
    self.end_effector_z_pub = rospy.Publisher("/robot/end_effector_z_estimated",Float64,queue_size=10)


    # KINEMATICS

    self.t1, self.t2, self.t3, self.t4 = sp.symbols("t1 t2 t3 t4")
    self.joint_angle_vars = sp.Matrix([self.t1, self.t2, self.t3, self.t4])

    A10 = sp.Matrix([
        [-sp.sin(self.t1), 0, sp.cos(self.t1), 0],
        [sp.cos(self.t1), 0, sp.sin(self.t1), 0],
        [0,1,0,2.5],
        [0,0,0,1]
    ])

    A21 = sp.Matrix([
        [-sp.sin(self.t2), 0, sp.cos(self.t2), 0],
        [sp.cos(self.t2), 0, sp.sin(self.t2), 0],
        [0, 1, 0, 0],
        [0,0,0,1]
    ])

    A32 = sp.Matrix([
        [sp.cos(self.t3), 0, -sp.sin(self.t3), 3.5*sp.cos(self.t3)],
        [sp.sin(self.t3), 0, sp.cos(self.t3), 3.5*sp.sin(self.t3)],
        [0, -1, 0, 0],
        [0,0,0,1]
    ])
        
    A43 = sp.Matrix([
        [sp.cos(self.t4), -sp.sin(self.t4), 0, 3*sp.cos(self.t4)],
        [sp.sin(self.t4), sp.cos(self.t4), 0, 3*sp.sin(self.t4)],
        [0, 0, 1, 0],
        [0,0,0,1]
    ])

    self.homogenous_transformation_matrix = A10*A21*A32*A43
    self.error = np.array([0.0,0.0,0.0], dtype='float64')  
    self.error_d = np.array([0.0,0.0,0.0], dtype='float64') 

    self.time_previous_step = np.array([rospy.get_time()], dtype='float64')

    self.joint_angles = np.array([0.0,0.0,0.0,0.0])
    self.target = np.array([0.0,0.0,0.0])
  
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
    #contours, _ = cv2.findContours(green_mask_image2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    M_image2 = cv2.moments(green_mask_image2)
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

    if (M_YZ['m00'] != 0):
      cy = int(M_YZ['m10']/M_YZ['m00'])
      cz = int(M_YZ['m01']/M_YZ['m00'])
    else:
      cy = 0
      cz = 0

    M_XZ = cv2.moments(sphere_XZ)

    if (M_XZ['m00'] != 0):
      cx = int(M_XZ['m10']/M_XZ['m00'])
      cz = int(M_XZ['m01']/M_XZ['m00']) 
    else:
      cx = 0
      cz = 0

    pos = np.array([cx,cy,cz])
    pos = 6 * pos/np.linalg.norm(pos)

    return pos

  def p2mBlue(self,image1,image2):
    blue = self.detect_blue(image1, image2)
    yellow = self.detect_yellow(image1, image2)
    v = blue - yellow
    return 2.5 * v/np.linalg.norm(v)

  def p2mGreen(self,image1,image2):
    green = self.detect_green(image1, image2)
    blue = self.detect_blue(image1, image2)
    v = green - blue
    return 3.5 * v/np.linalg.norm(v) + self.p2mBlue(image1, image2)

  def p2mRed(self,image1,image2):
    red = self.detect_red(image1, image2)
    green = self.detect_green(image1, image2)
    v = red - green
    return 3 * v/np.linalg.norm(v) + self.p2mGreen(image1, image2)

  # Joint angle 2
  def detect_joint_angle2(self, image1, image2):

    # Detection problem with images- should be done with image1 ONLY
    blueBlob = self.detect_blue(image1, image2)
    greenBlob = self.detect_green(image1, image2)

    joint2 = greenBlob - blueBlob

    # Create an axis perpendicular to rotation axis- X
    normToXAxis = [0, 0, -1]

    dotProduct = normToXAxis[0] * joint2[0] + normToXAxis[1] * joint2[1]  + normToXAxis[2] * joint2[2]

    normalizedVector1 = np.sqrt(normToXAxis[0]**2+normToXAxis[1]**2+normToXAxis[2]**2)
    normalizedVector2 = np.sqrt(joint2[0]**2+joint2[1]**2+joint2[2]**2) 

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

  def estimateJointAngles(self, image1, image2):
    ja2 = self.detect_joint_angle2(image1, image2)
    ja3 = self.detect_joint_angle3(image1, image2)
    ja4 = self.detect_joint_angle4(image1, image2)
    return np.array([0.0,ja2,ja3,ja4])

  def forwardKinematics(self,theta1,theta2,theta3,theta4):
    position = self.homogenous_transformation_matrix[:-1,-1].evalf(subs = {
            self.t1 : theta1,
            self.t2 : theta2,
            self.t3 : theta3,
            self.t4 : theta4
        })
    return np.array(position).astype(np.float64).flatten()

  def calculateJacobian(self,q):
    [ja1, ja2, ja3, ja4] = q
    jacobian = self.homogenous_transformation_matrix[:-1,-1].jacobian(self.joint_angle_vars)
    jacobian = jacobian.evalf(subs = {
      self.t1 : ja1,
      self.t2 : ja2,
      self.t3 : ja3,
      self.t4 : ja4
    })
    J = np.array(jacobian).astype(np.float64)
    return J

  def pd_control(self,image1,image2):
    K_p = np.array([[3.0,0.0,0.0], [0.0,3.0,0.0],[0.0,0.0,3.0]])
    K_d = np.array([[0.1,0.0,0.0], [0.0,0.1,0.0],[0.0,0.0,0.1]])
    curr_time = np.array([rospy.get_time()])
    dt = curr_time - self.time_previous_step
    self.time_previous_step = curr_time
    # end_effector_pos = self.p2mRed(image1, image2) * np.array([1,1,-1]) <- This works perfectly as well as the forward kinematics!
    q = self.joint_angles
    end_effector_pos = self.forwardKinematics(*self.joint_angles)
    xe = Float64(data = end_effector_pos[0])
    ye = Float64(data = end_effector_pos[1])
    ze = Float64(data = end_effector_pos[2])
    self.end_effector_x_pub.publish(xe)
    self.end_effector_y_pub.publish(ye)
    self.end_effector_z_pub.publish(ze)
    print(xe.data)
    desired_pos = self.detect_orange_sphere(image1,image2) * np.array([1,1,-1])
    # desired_pos = self.target
    self.error_d = ((desired_pos - end_effector_pos) - self.error)/dt
    # estimate error
    self.error = desired_pos-end_effector_pos
    # q = self.estimateJointAngles(image1, image2) # estimate initial value of joint angles
    J_inv = np.linalg.pinv(self.calculateJacobian(q))  # calculating the psudeo inverse of Jacobian
    dq_d =np.dot(J_inv, ( np.dot(K_d,self.error_d.transpose()) + np.dot(K_p,self.error.transpose()) ) )  # control input (angular velocity of joints)
    q_d = q + (dt * dq_d)  # control input (angular position of joints)
    return q_d


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
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
  
  # Recieve data from camera 2, process it, and publish
  def callback2(self, data):

    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data,"bgr8")

    except CvBridgeError as e:
      print(e)

    ### NOTE: Each section should be run separately

    # # # SECTION 2.1:
    
    # ACTUAL VALUES
    joint2Value = Float64()
    joint2Value.data = ((np.pi/2) * np.sin((np.pi/15) * rospy.get_time()))
    self.joint2_pub.publish(joint2Value)

    joint3Value = Float64()
    joint3Value.data = ((np.pi/2) * np.sin((np.pi/18) * rospy.get_time()))
    self.joint3_pub.publish(joint3Value)

    joint4Value = Float64()
    joint4Value.data = ((np.pi/2) * np.sin((np.pi/20) * rospy.get_time()))
    self.joint4_pub.publish(joint4Value)
      
    # # # MY ESTIMATED VALUES
    joint2EstimatedValue = Float64()
    joint2EstimatedValue.data = self.detect_joint_angle2(self.cv_image1, self.cv_image2)
    self.joint2_estimate_pub.publish(joint2EstimatedValue)

    joint3EstimatedValue = Float64()
    joint3EstimatedValue.data = self.detect_joint_angle3(self.cv_image1, self.cv_image2)
    self.joint3_estimate_pub.publish(joint3EstimatedValue)

    joint4EstimatedValue = Float64()
    joint4EstimatedValue.data = self.detect_joint_angle4(self.cv_image1, self.cv_image2)
    self.joint4_estimate_pub.publish(joint4EstimatedValue)

    # # # DIFFERENCES BETWEEN ACTUAL VALUES AND ESTIMATED VALUES- uncomment for verification if needed
    # print("Differences between Joint2 actual and estimated joint angle values:")
    # print(abs(joint2Value.data - joint2EstimatedValue.data))

    # print("Differences between Joint3 actual and estimated joint angle values:")
    # print(abs(joint3Value.data - joint3EstimatedValue.data))

    # print("Differences between Joint4 actual and estimated joint angle values:")
    # print(abs(joint4Value.data - joint4EstimatedValue.data))

    ## SECTION 2.2:
    # ESTIMATED VALUES
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
    # im1=cv2.imshow('window1', self.cv_image1)
    # im2=cv2.imshow('window2', self.cv_image2)
    # cv2.waitKey(1)

    #Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
    except CvBridgeError as e:
      print(e)

    # SECTION 3.2 ***************** Uncomment to run
    # new_joint_angles = self.pd_control(self.cv_image1, self.cv_image2)
    # self.joint1=Float64()
    # self.joint1.data= new_joint_angles[0]
    # self.joint2=Float64()
    # self.joint2.data= new_joint_angles[1]
    # self.joint3=Float64()
    # self.joint3.data= new_joint_angles[2]
    # self.joint4=Float64()
    # self.joint4.data= new_joint_angles[3]

    # self.joint1_pub.publish(self.joint1)
    # self.joint2_pub.publish(self.joint2)
    # self.joint3_pub.publish(self.joint3)
    # self.joint4_pub.publish(self.joint4)

    # SECTION 3.1 *****************
    # joint_angles = np.array([0.0, joint2Value.data, 0.0, 0.0])
    # fk = self.forwardKinematics(*joint_angles).flatten()
    # cv = self.p2mRed(self.cv_image1, self.cv_image2)
    # print("FK",fk)
    # print("CV",cv)
    # print("Error", np.linalg.norm(fk-cv))
    # print()

  def updateAngles(self,msg):
    self.joint_angles = msg.position

  def targetx(self,x):
    self.target[0] = x.data

  def targety(self,x):
    self.target[1] = x.data

  def targetz(self,x):
    self.target[2] = x.data



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