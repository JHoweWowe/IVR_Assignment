#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import sympy as sp
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64


class Kinematics:

    def __init__(self):

        rospy.init_node('kinematics', anonymous=True)

        self.angle1_actual = rospy.Subscriber("/robot/joint1_position_controller/command",Float64,self.callback1)
        self.angle2_actual = rospy.Subscriber("/robot/joint2_position_controller/command",Float64,self.callback2)
        self.angle3_actual = rospy.Subscriber("/robot/joint3_position_controller/command",Float64,self.callback3)
        self.angle4_actual = rospy.Subscriber("/robot/joint4_position_controller/command",Float64,self.callback4)

        self.t1, self.t2, self.t3, self.t4 = sp.symbols("t1 t2 t3 t4")
        joint_angles = sp.Matrix([self.t1, self.t2, self.t3, self.t4])

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
        

    def forwardKinematics(self,theta1,theta2,theta3,theta4):
        position = self.homogenous_transformation_matrix[:1,-1].evalf({
            self.t1 : theta1,
            self.t2 : theta2,
            self.t3 : theta3,
            self.t4 : theta4
        })
        return position


    def callback1(self,data):
        print(data)

    def callback2(self,data):
        print(data)

    def callback3(self,data):
        print(data)a

    def callback4(self,data):
        print(data)

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
