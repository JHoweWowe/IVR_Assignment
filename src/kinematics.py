#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

class Kinematics:

    def __init__(self):

        rospy.init_node('kinematics', anonymous=True)

        self.angle1_actual = rospy.Subscriber("/robot/joint1_position_controller/command",Float64,self.callback1)
        self.angle2_actual = rospy.Subscriber("/robot/joint2_position_controller/command",Float64,self.callback2)
        self.angle3_actual = rospy.Subscriber("/robot/joint3_position_controller/command",Float64,self.callback3)
        self.angle4_actual = rospy.Subscriber("/robot/joint4_position_controller/command",Float64,self.callback4)

        self.end_effector_cv = rospy.Subscriber("robot/end_effector",numpy_msg(Floats),self.compareCVFK)

        self.joint_array = [0.0,0.0,0.0,0.0]

    def forwardKinematics(self,theta1,theta2,theta3,theta4):
        A10 = np.array([[-np.sin(theta1), 0, np.cos(theta1), 0],
        [np.cos(theta1), 0, np.sin(theta1), 0],
        [0, 1, 0, 2.5],
        [0,0,0,1]])
        A21 = np.array([[-np.sin(theta2), 0, np.cos(theta2), 0],
        [np.cos(theta2), 0, np.sin(theta2), 0],
        [0, 1, 0, 0],
        [0,0,0,1]])
        A32 = np.array([[np.cos(theta3), 0, -np.sin(theta3), 3.5*np.cos(theta3)],
        [np.sin(theta3), 0, np.cos(theta3), 3.5*np.sin(theta3)],
        [0, -1, 0, 0],
        [0,0,0,1]])
        A43 = np.array([[np.cos(theta4), -np.sin(theta4), 0, 3*np.cos(theta4)],
        [np.sin(theta4), np.cos(theta4), 0, 3*np.sin(theta4)],
        [0, 0, 1, 0],
        [0,0,0,1]])
        htm = np.dot(A10,np.dot(A21,np.dot(A32,A43)))
        return htm[:3,-1]


    def callback1(self,data):
        self.joint_array[0] = data

    def callback2(self,data):
        self.joint_array[1] = data

    def callback3(self,data):
        self.joint_array[2] = data

    def callback4(self,data):
        self.joint_array[3] = data

    def compareCVFK(self,data):
        effectorPositionFK = self.forwardKinematics(*self.joint_array)
        print(data, effectorPositionFK)

# call the class
def main(args):
  k = Kinematics()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
