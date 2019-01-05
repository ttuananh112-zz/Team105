import rospy
from std_msgs.msg import Float32
import cv2
import math
import time
from time import sleep

class car_control:
    def __init__(self, team_number):
        self.speed_pub = rospy.Publisher("Team"+team_number+"_speed", Float32, queue_size=1)
        self.steerAngle_pub = rospy.Publisher("Team"+team_number+"_steerAngle", Float32, queue_size=1)
        rospy.init_node('cds', anonymous = True)
        rospy.Rate(10)
        self.last_detected = 0
        self.sign_type = 0

    def control(self, signWidth, middlePos):
        if not rospy.is_shutdown():
            steerAngle = self.cal_steerAngle(signWidth, middlePos)
            if math.fabs(steerAngle) >= 10 or signWidth > 35 or signWidth < -35:
                self.speed_pub.publish(35)
            else:
                self.speed_pub.publish(55)

    def cal_steerAngle(self, signWidth, middlePos):
        carPos_x = 160
        carPos_y = 240

        middlePos_x = middlePos[0]
        middlePos_y = middlePos[1]

        steerAngle = 0
        now = time.time()
        if (signWidth > 35):
            self.last_detected = time.time()
            self.sign_type = 1
        if (signWidth < -35):
            self.last_detected = time.time()
            self.sign_type = -1

        diff = now - self.last_detected
        if (diff > 0.3 and diff < 1.5):
            steerAngle = 20 * self.sign_type
            self.steerAngle_pub.publish(steerAngle)
        else:
            # print("STRAIGHT")
            # Can't detect lane
            if middlePos_x == -1:
                steerAngle = 0
            else:
                # Distance between MiddlePos and CarPos
                distance_x = middlePos_x - carPos_x
                distance_y = carPos_y - middlePos_y

                # Angle to middle position
                steerAngle = math.atan(distance_x / distance_y) * 180 / math.pi
                # print(middlePos_x, steerAngle)
            self.steerAngle_pub.publish(steerAngle)

        return steerAngle



