import numpy as np
import rospy
from IPython import embed

from std_msgs.msg import Float64MultiArray

TRANSLATIONAL_PUBLISHER_TOPIC = '/translation_tensor'
ROTATIONAL_PUBLISHER_TOPIC = '/rotational_tensor'
GRIPPER_PUBLISHER_TOPIC = '/gripper_tensor'

class TensorSubscriber (object):
    def __init__(self):
        try:
            rospy.init_node('tensor_receiver')
        except:
            pass

        rospy.Subscriber(TRANSLATIONAL_PUBLISHER_TOPIC, Float64MultiArray, self._callback_translation_data, queue_size=1)
        rospy.Subscriber(ROTATIONAL_PUBLISHER_TOPIC, Float64MultiArray, self._callback_rotation_data, queue_size=1)
        rospy.Subscriber(GRIPPER_PUBLISHER_TOPIC, Float64MultiArray, self._callback_gripper_data, queue_size=1)

        self.translation_tensor = None
        self.rotational_tensor = None
        self.gripper_tensor = None
        self.tr_data_offset = None
        self.rot_data_offset = None
        self.gr_data_offset = None

    def _callback_translation_data(self, data):
        self.translation_tensor = list(data.data)
        self.tr_data_offset = data.layout.data_offset

    def _callback_rotation_data(self, data):
        self.rotational_tensor = list(data.data)
        self.rot_data_offset = data.layout.data_offset

    def _callback_gripper_data(self, data):
        self.gripper_tensor = list(data.data)    
        self.gr_data_offset = data.layout.data_offset
