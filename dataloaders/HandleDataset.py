import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import glob
from tqdm import tqdm
import json
from PIL import Image, ImageFile
import random
from collections import defaultdict
ImageFile.LOAD_TRUNCATED_IMAGES = True

class HandleDataset(Dataset):

    def __init__(self, params, encoder, partial=None):
        self.params = params
        self.encoder = encoder

        self.img_tensors = []
        self.representations = []
        self.translation = []
        self.rotation = []
        self.gripper = []
        self.paths = []
        self.path_dict = defaultdict(list)
        self.frame_index = defaultdict(int)

        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Resize((self.params.img_size, self.params.img_size)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        self.extract_data(partial)

    def extract_data(self, factor=None):
        index = 0
        runs = glob.glob(self.params['folder']+'/*')
        random.shuffle(runs)
        total = len(runs)
        if factor is not None:
            total = int(total * factor)

        for run_index in tqdm(range(total)):
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

                if(self.params['representation'] == 1):
                    self.img_tensors.append(img_tensor.detach())
                    if(self.params['bc_model'] == 'BC_Full'):
                        self.translation.append(torch.FloatTensor(action_dict[frame][0:3]))
                        self.rotation.append(torch.FloatTensor(action_dict[frame][3:6]))
                        self.gripper.append(torch.tensor([action_dict[frame][6]]))
                        self.paths.append(runs[run_index]+'/'+frame)
                        self.path_dict[runs[run_index]].append(frame)
                        self.frame_index[runs[run_index] + '/' + frame] = index

                else:
                    represnetation = self.encoder.encode(img_tensor)[0]
                    self.representations.append(represnetation.detach())
                    self.translation.append(torch.FloatTensor(action_dict[frame][0:3]))
                    self.rotation.append(torch.FloatTensor(action_dict[frame][3:6]))
                    self.gripper.append(torch.tensor([action_dict[frame][6]]))
                    self.paths.append(runs[run_index]+'/'+frame)
                    self.path_dict[runs[run_index]].append(frame)
                    self.frame_index[runs[run_index] + '/' + frame] = index
                index += 1

    def get_subset(self, factor):
        total = int(factor * len(self.path_dict))
        keys = random.sample(list(self.path_dict.keys()), total)
        items = []
        for k in keys:
            for v in self.path_dict[k]:
                items.append(self.__getitem__(self.frame_index[k + '/' +  v]))
        return items


    def __len__(self):
        return(max(len(self.img_tensors),len(self.representations)))

    def __getitem__(self, index):
        if(self.params['representation'] == 1):
            if(self.params['bc_model'] == 'BC_Full'):
                return((self.img_tensors[index], self.translation[index], self.rotation[index], self.gripper[index], self.paths[index]))
            else:
                return(self.img_tensors[index])
        else:
            return((self.representations[index], self.translation[index], self.rotation[index], self.gripper[index], self.paths[index]))
