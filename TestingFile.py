from ServoHatModule import *
import time


pwm = PCA9685(0x40, debug=False)
pwm.setPWMFreq(50)
pwm.setServoPulse(1, 1500)
pwm.setServoPulse(0, 1500)

time.sleep(2)
pwm.setServoPulse(1, 500)

# import cv2
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     s, d = cap.read()
#     cv2.imshow('test', d)
#     key = cv2.waitKey(1)
#     if key == 13:
#         break

