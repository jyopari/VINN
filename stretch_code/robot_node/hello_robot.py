import stretch_body.robot
import numpy as np
import PyKDL
import rospy
from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R
import math
import time
import random


joint_list = ["joint_fake","joint_lift","joint_arm_l3","joint_arm_l2","joint_arm_l1" ,"joint_arm_l0","joint_wrist_yaw","joint_wrist_pitch","joint_wrist_roll"]

class HelloRobot:
    def __init__(self):
        #Initializing ROS node
        try:
            rospy.init_node('hello_robot_node')
        except:
            pass

        # Creating a robot object
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()
        # Initializing the robot
        self.base_x = self.robot.base.status['x']
        self.base_y = self.robot.base.status['y']
        self.robot.base.translate_by(0.05)
        self.robot.push_command()
        time.sleep(2)

        # Configuring the forward directions
        fwd_x = self.robot.base.status['x']
        fwd_y = self.robot.base.status['y']
        self.far_x = (fwd_x - self.base_x)*100 + self.base_x
        self.far_y = (fwd_y - self.base_y)*100 + self.base_y
        self.robot.base.translate_by(-0.05)
        self.far_to_origin = math.sqrt((self.base_y - self.far_y)**2 + (self.base_x - self.far_x)**2)
        self.robot.push_command()

        #trash, recycle       
        
        arm_start_pos = (24+(random.random()-.5)*10)/100.0
        print(arm_start_pos)
        self.robot.arm.move_to(arm_start_pos)
        self.robot.lift.move_to(0.86)
        
        #other type
        '''
        arm_start_pos = (0 + (random.random()-.5)*5)/100.0
        self.robot.arm.move_to(arm_start_pos)
        self.robot.lift.move_to(.87)
        '''   

        self.robot.end_of_arm.move_to('wrist_yaw',1.53)
        self.robot.end_of_arm.move_to('wrist_pitch',-0.01)
        self.robot.end_of_arm.move_to('wrist_roll',0)
        self.robot.end_of_arm.move_to('stretch_gripper',40)
        self.robot.push_command()

        time.sleep(2)
        # Constraining the robots movement
        self.clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

        # Joint dictionary for Kinematics
        self.joints = {'joint_fake':0}

        robot_model = URDF.from_xml_file('/home/hello-robot/robot-files/stretch_nobase_raised.urdf')
    
        kdl_tree = kdl_tree_from_urdf_model(robot_model)
        self.arm_chain = kdl_tree.getChain('base_link', 'link_raised_gripper')

        self.joint_array = PyKDL.JntArray(self.arm_chain.getNrOfJoints())

        # Forward kinematics
        self.fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self.arm_chain)

        # Inverse Kinematics
        self.ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self.arm_chain)
        self.ik_p_kdl = PyKDL.ChainIkSolverPos_NR(self.arm_chain, self.fk_p_kdl, self.ik_v_kdl) 

    def updateJoints(self):
        origin_dist = math.sqrt((self.base_y - self.robot.base.status['y'])**2+(self.base_x - self.robot.base.status['x'])**2)
        far_dist = math.sqrt((self.far_y - self.robot.base.status['y'])**2+(self.far_x - self.robot.base.status['x'])**2)
        
        if(far_dist <= self.far_to_origin):
            self.joints['joint_fake'] = origin_dist
        else:
            self.joints['joints_fake'] = -1*origin_dist
        
        self.joints['joint_lift'] = self.robot.lift.status['pos']
        
        armPos = self.robot.arm.status['pos']
        self.joints['joint_arm_l3'] = armPos / 4.0
        self.joints['joint_arm_l2'] = armPos / 4.0
        self.joints['joint_arm_l1'] = armPos / 4.0
        self.joints['joint_arm_l0'] = armPos / 4.0
        
        self.joints['joint_wrist_yaw'] = self.robot.end_of_arm.status['wrist_yaw']['pos']
        self.joints['joint_wrist_roll'] = self.robot.end_of_arm.status['wrist_roll']['pos']
        self.joints['joint_wrist_pitch'] = self.robot.end_of_arm.status['wrist_pitch']['pos']
        self.joints['joint_stretch_gripper'] = self.robot.end_of_arm.status['stretch_gripper']['pos']  


    def move(self, ik_joints, gripper):
        self.robot.base.translate_by(ik_joints['joint_fake']-self.joints['joint_fake'], 5)
        self.robot.arm.move_to(  ik_joints['joint_arm_l3'] + 
                            ik_joints['joint_arm_l2'] + 
                            ik_joints['joint_arm_l1'] + 
                            ik_joints['joint_arm_l0'])
        
        self.robot.lift.move_to(ik_joints['joint_lift'])
        #old vals : .2 , 2.1

        if(gripper[0] < 0):
            self.robot.end_of_arm.move_to('stretch_gripper', -30)
        if(gripper[0] > 2.1):
            self.robot.end_of_arm.move_to('stretch_gripper', 40)

        self.robot.push_command()

        time.sleep(.5)
    
    def move_to_pose(self, translation_tensor, rotational_tensor, gripper):
        # Correcting for the camera tilt
        translation = [translation_tensor[0], 
                       translation_tensor[1] + np.sin(0.349066)*translation_tensor[2], 
                       np.cos(0.349066)*translation_tensor[2]-np.sin(0.349066)*translation_tensor[1]]
        rotation = rotational_tensor

        # move logic
        self.updateJoints()
        for joint_index in range(self.joint_array.rows()):
            self.joint_array[joint_index] = self.joints[joint_list[joint_index]]

        curr_pose = PyKDL.Frame()
        goal_pose = PyKDL.Frame()
        self.fk_p_kdl.JntToCart(self.joint_array, curr_pose)

        curr_rot = R.from_quat(curr_pose.M.GetQuaternion()).as_dcm()
        # Ignore rotation for now
        # rot_goal_m = curr_rot.as_dcm().dot(R.from_euler('xyz', rotation, degrees=False).as_dcm())
        rot_matrix = R.from_euler('xyz', rotation, degrees=False).as_dcm()

        curr_rot[:, [1,2]] = curr_rot[:, [2,1]]
        curr_rot[:, [0,2]] = curr_rot[:, [2,0]]

        rot_matrix[:, [0,1]] = rot_matrix[:, [1,0]]
        rot_matrix[:, [0,2]] = rot_matrix[:, [2,0]]

        rot_goal_m = curr_rot.dot(rot_matrix)


        rot_goal = PyKDL.Rotation(PyKDL.Vector(rot_goal_m[0][0], rot_goal_m[1][0], rot_goal_m[2][0]),
                                  PyKDL.Vector(rot_goal_m[0][1], rot_goal_m[1][1], rot_goal_m[2][1]),
                                  PyKDL.Vector(rot_goal_m[0][2], rot_goal_m[1][2], rot_goal_m[2][2]))
        scale_side = 0.07
        scale_fwd = 0.07

        goal_pose.p[0] = curr_pose.p[0] + scale_fwd*translation[1]
        goal_pose.p[1] = curr_pose.p[1] - scale_side*translation[0]
        goal_pose.p[2] = curr_pose.p[2] + 0*translation[2]

        goal_pose.M = rot_goal

        seed_array = PyKDL.JntArray(self.arm_chain.getNrOfJoints())
        self.ik_p_kdl.CartToJnt(seed_array, goal_pose, self.joint_array)

        ik_joints = {}

        for joint_index in range(self.joint_array.rows()):
            ik_joints[joint_list[joint_index]] = self.joint_array[joint_index]

        self.move(ik_joints, gripper)
