import rospy
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.optim as optimi
import torchvision.transforms as T

import pickle

import cv2

from sensor_msgs.msg import Image as Image_msg
from std_msgs.msg import Float64MultiArray, Int64
from cv_bridge import CvBridge, CvBridgeError

import glob
from tqdm import tqdm
from PIL import Image 
from copy import deepcopy as copy
import json
import pickle
import matplotlib.pyplot as plt

IMAGE_SUBSCRIBER_TOPIC = '/gopro_image'

TRANSLATIONAL_PUBLISHER_TOPIC = '/translation_tensor'
ROTATIONAL_PUBLISHER_TOPIC = '/rotational_tensor'
GRIPPER_PUBLISHER_TOPIC = '/gripper_tensor'

PING_TOPIC = 'run_model_ping'

class Identity(nn.Module):
    '''
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    '''
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#BC

class forward_model_translation(nn.Module):
    def __init__(self):
        super(forward_model_translation, self).__init__()

        self.f1 = nn.Linear(2048, 2048)
        self.f2 = nn.Linear(2048, 1024)
        self.f3 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        return(x)

class forward_model_rotation(nn.Module):
    def __init__(self):
        super(forward_model_rotation, self).__init__()

        self.f1 = nn.Linear(2048, 2048)
        self.f2 = nn.Linear(2048, 1024)
        self.f3 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        return(x)

class forward_model_gripper(nn.Module):
    def __init__(self):
        super(forward_model_gripper, self).__init__()

        self.f1 = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.f1(x)
        return(x)

# VAE

class VAE_Translation(nn.Module):
    def __init__(self, input_dim):
        super(VAE_Translation, self).__init__()
        self.e1 = nn.Linear(input_dim, input_dim)
        self.e2 = nn.Linear(input_dim, 1024)
        self.e3 = nn.Linear(1024, 128)
        self.e_mu = nn.Linear(128,128)
        self.e_sigma = nn.Linear(128,128)

        self.d1 = nn.Linear(128, 128)
        self.d2 = nn.Linear(128, 3)

    def encode(self, x):
        x = self.e1(x)
        x = F.relu(x)
        x = self.e2(x)
        x = F.relu(x)
        x = self.e3(x)
        x = F.relu(x)
        mu = self.e_mu(x)
        sigma = self.e_sigma(x)
        return((mu,sigma))

    def decode(self, x):
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)
        return(x)

	# From https://github.com/pytorch/examples/blob/master/vae/main.py
    def reparameterize(self,mu,sigma):
        std = torch.exp(0.5*sigma)
        eps = torch.randn_like(std)
        return (mu + eps*std)

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu,sigma)
        action = self.decode(z)
        return((action, mu, sigma))


class getOutput (object):
    def __init__ (self):
        # Initializing a rosnode
        try:
            rospy.init_node('image_subscriber')       
        except:
            pass


        self.preprocess = T.Compose([T.ToTensor(),
                                T.Resize((224,224)),
                                T.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

        self.softmax = torch.nn.Softmax(dim=0)

        self.nn_counter = 0

        # Getting images from the rostopic
        self.image = None
        self.bridge = CvBridge()

        # Subscriber for images
        rospy.Subscriber(IMAGE_SUBSCRIBER_TOPIC, Image_msg, self._callback_image, queue_size=1)
        rospy.Subscriber(PING_TOPIC, Int64, self._callback_ping, queue_size=1)
        self.uid = -1

        # Publishers for the evaluated tensors
        self.translational_publisher = rospy.Publisher(TRANSLATIONAL_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1)
        self.rotational_publisher = rospy.Publisher(ROTATIONAL_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1)
        self.gripper_publisher = rospy.Publisher(GRIPPER_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1)

        self.translational_publisher = rospy.Publisher(TRANSLATIONAL_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1)

        # Initializing the models
        
        self.resnet = models.resnet50(pretrained=False)
        self.rotation_model = forward_model_rotation()

        #BC
        self.translation_model = forward_model_translation()
        self.gripper_model = forward_model_gripper()

        # Getting the translation parameters
        translation_state_dict = torch.load('model_weights/HandleModel_translation_all_8000.pt', map_location=torch.device('cpu'))
        self.translation_model.load_state_dict(translation_state_dict['model_state_dict'])

        # Getting the rotation parameters
        rotation_state_dict = torch.load('model_weights/HandleModel_rotation_7900.pt', map_location=torch.device('cpu'))
        self.rotation_model.load_state_dict(rotation_state_dict['model_state_dict'])
        
        # Getting the gripper parameters
        gripper_state_dict = torch.load('model_weights/HandleModel_gripper_all_8000.pt', map_location=torch.device('cpu'))
        self.gripper_model.load_state_dict(gripper_state_dict['model_state_dict'])        

        # Getting the resnet parameters
        resnet_state_dict = torch.load('model_weights/BYOL_100_handle_all.pt', map_location=torch.device('cpu'))
        self.resnet.load_state_dict(resnet_state_dict['model_state_dict'])
        self.resnet.fc = Identity()


    def _callback_image(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        except CvBridgeError as e:
            print(e)

    def _callback_ping(self, data):
        self.uid = int(data.data)
        print('Received uid {}'.format(self.uid))

    def dist_metric(self,x,y):
        return(torch.norm(x-y).item())

    def calculate_action_translation(self,dist_list,k):
        action = torch.tensor([0.0,0.0,0.0])
        top_k_weights = torch.zeros((k,))
        for i in range(k):
            top_k_weights[i] = dist_list[i][0]
            
            im = Image.open(dist_list[i][2])
            im.save('nn_og/'+str(self.nn_counter)+'_'+str(i)+'.jpg')
            action_img = cv2.imread('nn_og/'+str(self.nn_counter)+'_'+str(i)+'.jpg')
            start_point = (1920//2, 1080//2)
            scale = 400
            x = np.array([dist_list[i][1][0], dist_list[i][1][1]])
            x = x / np.linalg.norm(x)
            end_point = (int(1920//2 + x[0]  *scale), int(1080//2 - x[1]*scale))
            
            img = cv2.arrowedLine(action_img, start_point, end_point, (0,0,255), 4)
            cv2.imwrite('nn_action/'+str(self.nn_counter)+'_'+str(i)+'.jpg', img)
            im.close()
    
        top_k_weights = self.softmax(-1*top_k_weights)
        for i in range(k):
            action = torch.add(top_k_weights[i]*dist_list[i][1], action)
        
        self.nn_counter += 1
        
        return(action)

    def calculate_action_gripper(self,dist_list,k):
        action = torch.tensor([0.0])
        top_k_weights = torch.zeros((k,))
        for i in range(k):
            top_k_weights[i] = dist_list[i][0]

        top_k_weights = self.softmax(-1*top_k_weights)
        for i in range(k):
            action = torch.add(top_k_weights[i]*dist_list[i][1][0], action)

        return(action)

    def extract_data_temporal(self, t):
        runs = glob.glob('right/*')
        dataset = []

        for run_index in tqdm(range(len(runs))):
            run = runs[run_index]
            action_file = open(run+'/labels.json', 'r')
            action_dict = json.load(action_file)

            temporal_representations = []
            temporal_translation = []
            temporal_rotation = []
            temporal_gripper = []
            temporal_paths = []

            key_list = sorted(action_dict)
            for frame_index in range(len(key_list)):

                try:
                    img = Image.open(run+'/image_linear/'+key_list[frame_index])
                    img = img.crop((410, 0, 1600, 697))
                    img_tensor = self.preprocess(img)
                    img.close()
                except:
                    continue

                represnetation = self.resnet(img_tensor.reshape(1,3,224,224))[0]
                temporal_representations.append(represnetation.detach())
                temporal_translation.append(torch.FloatTensor(action_dict[key_list[frame_index]][0:3]))
                temporal_rotation.append(torch.FloatTensor(action_dict[key_list[frame_index]][3:6]))
                temporal_gripper.append(torch.tensor([action_dict[key_list[frame_index]][6]]))

                temporal_paths.append(runs[run_index]+'/images_linear/'+key_list[frame_index])

            if(len(temporal_representations) > t):
                for i in range(0, len(temporal_representations)-t):
                    curr_representation = torch.cat(tuple(temporal_representations[i:i+t+1]))
                    curr_translation = temporal_translation[i+t]
                    curr_rotation = temporal_rotation[i+t]
                    curr_gripper = temporal_gripper[i+t]
                    curr_path = temporal_paths[i+t]

                    dataset.append((curr_representation, curr_translation, curr_rotation, curr_gripper, curr_path))

        return(dataset)

    def extract_data(self):
        runs = glob.glob('train_all/*')
        dataset = []

        for run_index in tqdm(range(len(runs))):
            run = runs[run_index]
            action_file = open(run+'/labels.json', 'r')
            action_dict = json.load(action_file)

            for frame in action_dict:
                try:
                    img = Image.open(run+'/images_linear/'+frame)
                    img = img.crop((410, 0, 1600, 697))
                    img_tensor = self.preprocess(img)
                    img.close()
                except:
                    continue
                
                represnetation = self.resnet(img_tensor.reshape(1,3,224,224))[0].detach()
                translation = torch.FloatTensor(action_dict[frame][0:3])
                rotation = torch.FloatTensor(action_dict[frame][3:6])
                gripper = torch.tensor([action_dict[frame][6]])
                path = runs[run_index]+'/images_linear/'+frame
                dataset.append((represnetation,translation,rotation,gripper,path))

        return(dataset)

    def calculate_nearest_neighbors_gripper(self,img_tensor, dataset, k):
        dist_list = []

        for dataset_index in range(len(dataset)):

            dataset_embedding, dataset_translation, dataset_rotation, dataset_gripper, dataset_path = dataset[dataset_index]
            distance = self.dist_metric(img_tensor, dataset_embedding)
            dist_list.append((distance, dataset_gripper, dataset_path))

        dist_list = sorted(dist_list, key = lambda tup: tup[0])

        pred_action = self.calculate_action_gripper(dist_list, k)

        return(pred_action)

    def calculate_nearest_neighbors_translation(self,img_tensor, dataset, k):
        dist_list = []

        for dataset_index in range(len(dataset)):

            dataset_embedding, dataset_translation, dataset_rotation, dataset_gripper, dataset_path = dataset[dataset_index]
            distance = self.dist_metric(img_tensor, dataset_embedding)
            dist_list.append((distance, dataset_translation, dataset_path))

        dist_list = sorted(dist_list, key = lambda tup: tup[0])
        pred_action = self.calculate_action_translation(dist_list, k)
        print(dist_list[0:k])
        return(pred_action)

    def show_subscribed_image(self):
        rate = rospy.Rate(10)
        k = 5
        t = 0

        '''
        dataset = self.extract_data()
        

        with open('dataset_pickle.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        '''
        
        with open('dataset_pickle.pkl', 'rb') as f:
            dataset = pickle.load(f)
        

        print(dataset[0])
        prev_uid = -1
        window = []

        img_count = 0
        while True:

            rate.sleep()

            if self.image is None:
                continue

            if prev_uid == self.uid:
                continue
            
            print('different')
            prev_uid = copy(self.uid)

            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            im = Image.fromarray(img)
            #im = im.crop((2, 2, 632, 420))
            
            #save image
            im.save('robot_pov/'+str(img_count)+'.jpg')

            img_tensor = self.preprocess(im)
            im.close()

            embedding = self.resnet(img_tensor.reshape(1,3,224,224))[0]
          
            if(len(window) == 0):
                for i in range(t+1):
                    window.append(embedding)
            else:
                window.append(embedding)

            if(len(window) > t+1):
                window.pop(0)
            print(window)

            rotation_tensor = self.rotation_model(embedding).tolist() 

            translation_tensor = self.calculate_nearest_neighbors_translation(embedding,dataset,k).tolist()
            gripper_tensor = [self.calculate_nearest_neighbors_gripper(embedding,dataset,k).item()]
            rotation_tensor[0] = 0.0
            rotation_tensor[1] = 0.0
            rotation_tensor[2] = 0.0
            translation_tensor[2] = 0.0

            x = np.array([translation_tensor[0], translation_tensor[1]])
    
            og_img = cv2.imread('robot_pov/'+str(img_count)+'.jpg')

            start_point = (635//2, 380//2)
            scale = 300
            end_point = (int(635//2 + scale*x[0]), int(380//2 - scale*x[1]))
            action_img = cv2.arrowedLine(og_img, start_point, end_point, (0,0,255), 3)

            cv2.imwrite('robot_action/'+str(img_count)+'.jpg', action_img)

            img_count += 1
            print('translation', translation_tensor)
            print('gripper tensor', gripper_tensor)

            translation_publisher_list = Float64MultiArray()
            translation_publisher_list.layout.data_offset = self.uid
            translation_publisher_list.data = translation_tensor

            rotation_publisher_list = Float64MultiArray()
            rotation_publisher_list.layout.data_offset = self.uid
            rotation_publisher_list.data = rotation_tensor

            gripper_publisher_list = Float64MultiArray()
            gripper_publisher_list.layout.data_offset = self.uid
            gripper_publisher_list.data = gripper_tensor

            self.translational_publisher.publish(translation_publisher_list)
            self.rotational_publisher.publish(rotation_publisher_list)
            self.gripper_publisher.publish(gripper_publisher_list)


        cv2.destroyAllWindows()


if __name__ == '__main__':
    output = getOutput()
    output.show_subscribed_image()
