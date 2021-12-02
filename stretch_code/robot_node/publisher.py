import cv2
import rospy

import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

NODE_NAME = 'gopro_node'
IMAGE_PUBLISHER_NAME = '/gopro_image'

class ImagePublisher (object):
    def __init__(self):
        # Initializing camera
        self.camera = cv2.VideoCapture(3)

        # Initializing ROS node
        rospy.init_node(NODE_NAME)
        self.bridge = CvBridge()
        self.image_publisher = rospy.Publisher(IMAGE_PUBLISHER_NAME, Image, queue_size = 1)

    def publish_image_from_camera(self):
        rate = rospy.Rate(28)
        while True:
            # Getting the data from the camera
            ret_val, image = self.camera.read()
            image = image[100:480, 0:635]
            #cv2.imshow('my webcam', image)

            # Creating a CvBridge and publishing the data to the rostopic
            try:
                self.image_message = self.bridge.cv2_to_imgmsg(image, "bgr8")
            except CvBridgeError as e:
                print(e)

            self.image_publisher.publish(self.image_message)

            # Stopping the camera
            if cv2.waitKey(1) == 27:
                break

            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera_publisher = ImagePublisher()
    camera_publisher.publish_image_from_camera()
