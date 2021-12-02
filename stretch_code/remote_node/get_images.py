import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

IMAGE_SUBSCRIBER_TOPIC = '/gopro_image'

class ImageSubscriber (object):
    def __init__ (self):
        try:
            rospy.init_node('image_subscriber')       
        except:
            pass
        self.image = None
        self.bridge = CvBridge()
        rospy.Subscriber(IMAGE_SUBSCRIBER_TOPIC, Image, self._callback_image, queue_size=1)

    def _callback_image(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        except CvBridgeError as e:
            print(e)

    def show_subscribed_image(self):
        rate = rospy.Rate(15)
        while True:
            if self.image is None:
                continue
            cv2.imshow('Recieved GoPro Image', self.image)
            if cv2.waitKey(1) == 27:
                break
            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    image_subscriber = ImageSubscriber()
    print('Subscriber initialized!')
    image_subscriber.show_subscribed_image()
