#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('Team105')
import sys
import rospy
import cv2
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
from sign_detection import detect_sign
from sign_classi import predict
from lane_detector import lane_detector
from car_control import car_control

TEAM_NUMBER = '1'

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("Team" + TEAM_NUMBER + "_image/compressed", CompressedImage, callback=self.callback, queue_size=1)
        self.cc = car_control(TEAM_NUMBER)
        self.ld = lane_detector()
        rospy.Rate(10)

    def callback(self, data):
        # print("callback")
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # NOTE: image_np.shape = (240,320,3)
            sign_size = 0
            img, middlePos = self.ld.lane_detect(image_np)

            # out_img, sign_x, sign_y, sign_size = detect_sign(img)
            # cv2.imshow("Image window", img)
            # cv2.waitKey(1)

            # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            # print("Left ",left_fit," Right ",right_fit)
            cv2.imshow("Middle Pos", img)
            cv2.waitKey(1)

            # drive
            self.cc.control(sign_size, (middlePos[0], middlePos[2]))

        except CvBridgeError as e:
            print(e)



if __name__ == '__main__':
    rospy.init_node('cds', anonymous=True)
    # cv2.namedWindow("houghlines")
    # def nothing():
    #     pass
    # cv2.createTrackbar("rho", "houghlines",2,255,nothing)
    # cv2.createTrackbar("theta", "houghlines",180,255,nothing)
    # cv2.createTrackbar("minLine", "houghlines",78,255,nothing)
    # cv2.createTrackbar("maxGap", "houghlines",10,255,nothing)
    # cv2.waitKey(1)
    ic = image_converter()
    rospy.spin()
