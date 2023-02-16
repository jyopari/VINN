import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import wandb

class TranslationModel(nn.Module):
    def __init__(self, input_dim):
        super(TranslationModel, self).__init__()
        self.f1 = nn.Linear(input_dim, input_dim)
        self.f2 = nn.Linear(input_dim, 1024)
        self.f3 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        return(x)

class RotationModel(nn.Module):
    def __init__(self, input_dim):
        super(RotationModel, self).__init__()
        self.f1 = nn.Linear(input_dim, input_dim)
        self.f2 = nn.Linear(input_dim, 1024)
        self.f3 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        return(x)

class GripperModel(nn.Module):
    def __init__(self, input_dim):
        super(GripperModel, self).__init__()
        self.f1 = nn.Linear(input_dim, 4)

    def forward(self, x):
        return(self.f1(x))

class BC:
    def __init__(self, params):
        self.params = params
        self.params['representation'] = 0

        sys.path.append(params['root_dir'] + 'representation_models')
        sys.path.append(params['root_dir'] + 'dataloaders')
        from run_model import Encoder
        from HandleDataset import HandleDataset
        from PushDataset import PushDataset
        encoder = Encoder(params)

        if(self.params['wandb'] == 1):
            if(self.params['dataset'] == 'PushDataset'):
                wandb.init(project = 'Push BC', entity="nyu_vinn")
                wandb.run.name = 'Push_BC_' + str(self.params['pretrained'])
            if(self.params['dataset'] == 'StackDataset'):
                wandb.init(project = 'Stack BC', entity="nyu_vinn")
                wandb.run.name = 'Stack_BC_' + str(self.params['pretrained'])
            if(self.params['dataset'] == 'HandleData'):
                wandb.init(project = 'Handle BC', entity="nyu_vinn")
                wandb.run.name = 'Handle_BC_' + str(self.params['pretrained'])

        if(self.params['dataset'] == 'HandleData'):

            self.min_val_loss = float('inf')

            self.translation_loss_train = 0
            self.rotation_loss_train = 0
            self.gripper_loss_train = 0

            self.translation_loss_val = 0
            self.rotation_loss_val = 0
            self.gripper_loss_val = 0


            self.params['folder'] = self.params['train_dir']
            self.orig_img_data_train = HandleDataset(self.params, encoder)
            self.params['folder'] = self.params['val_dir']
            self.img_data_val = HandleDataset(self.params, encoder)

            if(self.params['gpu'] == 1):
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

            if(params['architecture'] == 'ResNet'):
                self.translation_model = TranslationModel(2048*(self.params['t']+1)).to(self.device)
                self.rotation_model = RotationModel(2048*(self.params['t']+1)).to(self.device)
                self.gripper_model = GripperModel(2048*(self.params['t']+1)).to(self.device)
            if(params['architecture'] == 'AlexNet'):
                self.translation_model = TranslationModel(9216*(self.params['t']+1)).to(self.device)
                self.rotation_model = RotationModel(9216*(self.params['t']+1)).to(self.device)
                self.gripper_model = GripperModel(9216*(self.params['t']+1)).to(self.device)

            self.optimizer = torch.optim.Adam(list(self.translation_model.parameters()) +
                                              list(self.rotation_model.parameters()) +
                                              list(self.gripper_model.parameters()), lr=self.params['lr'])

            self.dataLoader_val = DataLoader(self.img_data_val, batch_size=self.params['batch_size'], shuffle=True, pin_memory = True)

        if(self.params['dataset'] == 'PushDataset' or self.params['dataset'] == 'StackDataset'):
            self.min_val_loss = float('inf')
            self.min_test_loss = float('inf')

            self.translation_loss_train = 0
            self.translation_loss_val = 0
            self.translation_loss_test = 0

            self.params['folder'] = self.params['train_dir']
            self.orig_img_data_train = PushDataset(self.params, encoder)
            self.params['folder'] = self.params['val_dir']
            self.img_data_val = PushDataset(self.params, encoder)
            self.params['folder'] = self.params['test_dir']
            self.img_data_test = PushDataset(self.params, encoder)

            self.dataLoader_val = DataLoader(self.img_data_val, batch_size=self.params['batch_size'], shuffle=True, pin_memory = True)
            self.dataLoader_test = DataLoader(self.img_data_test, batch_size=self.params['batch_size'], shuffle=True, pin_memory = True)


    def train(self):
        if(self.params['gpu'] == 1):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        for epoch in tqdm(range(self.params['epochs'])):
            if(self.params['dataset'] == 'PushDataset' or self.params['dataset'] == 'StackDataset'):
                self.translation_loss_train = 0
                self.translation_loss_val = 0
                self.translation_loss_test = 0
            if(self.params['dataset'] == 'HandleData'):
                self.translation_loss_train = 0
                self.rotation_loss_train = 0
                self.gripper_loss_train = 0

                self.translation_loss_val = 0
                self.rotation_loss_val = 0
                self.gripper_loss_val = 0


            for i, data in enumerate(self.dataLoader_train, 0):

                self.optimizer.zero_grad()

                if(self.params['dataset'] == 'PushDataset' or self.params['dataset'] == 'StackDataset'):
                    representation, translation, path = data

                    pred_translation = self.translation_model(representation.float().to(self.device))
                    loss = self.mseLoss(pred_translation, translation.float().to(self.device))

                    self.translation_loss_train += loss.item() * representation.shape[0]

                if(self.params['dataset'] == 'HandleData'):

                    representation, translation, rotation, gripper, path = data

                    pred_translation = self.translation_model(representation.float().to(self.device))
                    pred_rotation = self.rotation_model(representation.float().to(self.device))
                    pred_gripper = self.gripper_model(representation.float().to(self.device))

                    translation_loss = self.mseLoss(pred_translation, translation.float().to(self.device))
                    rotation_loss = self.mseLoss(pred_rotation, rotation.float().to(self.device))
                    gripper_loss = self.ceLoss(pred_gripper,  gripper.reshape(pred_gripper.shape[0],).to(self.device))

                    self.translation_loss_train += translation_loss.item() * representation.shape[0]
                    self.rotation_loss_train += rotation_loss.item() * representation.shape[0]
                    self.gripper_loss_train += gripper_loss.item() * representation.shape[0]

                    loss = translation_loss + rotation_loss + gripper_loss

                loss.backward()
                self.optimizer.step()

            if(self.params['dataset'] == 'PushDataset' or self.params['dataset'] == 'StackDataset'):
                self.translation_loss_train /= len(self.img_data_train)
            if(self.params['dataset'] == 'HandleData'):
                self.translation_loss_train /= len(self.img_data_train)
                self.rotation_loss_train /= len(self.img_data_train)
                self.gripper_loss_train /= len(self.img_data_train)


            self.val()
            if(self.params['dataset'] == 'PushDataset' or self.params['dataset'] == 'StackDataset'):
                self.test()

            if(self.params['wandb'] == 1):
                self.wandb_publish()
            if(epoch%1000 == 0):
                self.save_model(epoch)

    def val(self):
        if(self.params['gpu'] == 1):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        for i, data in enumerate(self.dataLoader_val, 0):
            if(self.params['dataset'] == 'PushDataset' or self.params['dataset'] == 'StackDataset'):
                representation, translation, path = data

                if(self.params['gpu'] == 1):
                    pred_translation = self.translation_model(representation.float().to(self.device))
                    loss = self.mseLoss(pred_translation, translation.float().to(self.device))
                else:
                    pred_translation = self.translation_model(representation)
                    loss = self.mseLoss(pred_translation, translation)

                self.translation_loss_val += loss.item() * representation.shape[0]

            if(self.params['dataset'] == 'HandleData'):

                representation, translation, rotation, gripper, path = data

                pred_translation = self.translation_model(representation.float().to(self.device))
                pred_rotation = self.rotation_model(representation.float().to(self.device))
                pred_gripper = self.gripper_model(representation.float().to(self.device))

                translation_loss = self.mseLoss(pred_translation, translation.float().to(self.device))
                rotation_loss = self.mseLoss(pred_rotation, rotation.float().to(self.device))
                gripper_loss = self.ceLoss(pred_gripper,  gripper.reshape(pred_gripper.shape[0],).to(self.device))

                self.translation_loss_val += translation_loss.item() * representation.shape[0]
                self.rotation_loss_val += rotation_loss.item() * representation.shape[0]
                self.gripper_loss_val += gripper_loss.item() * representation.shape[0]

        if(self.params['dataset'] == 'PushDataset' or self.params['dataset'] == 'StackDataset'):
            self.translation_loss_val /= len(self.img_data_val)

            self.min_val_loss = min(self.min_val_loss, self.translation_loss_val)
            #print(self.min_val_loss)

        if(self.params['dataset'] == 'HandleData'):
            self.translation_loss_val /= len(self.img_data_val)
            self.rotation_loss_val /= len(self.img_data_val)
            self.gripper_loss_val /= len(self.img_data_val)

            self.min_val_loss = min(self.min_val_loss, self.translation_loss_val)
            #print(self.min_val_loss)

    def test(self):
        if(self.params['gpu'] == 1):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        for i, data in enumerate(self.dataLoader_test, 0):
            if(self.params['dataset'] == 'PushDataset' or self.params['dataset'] == 'StackDataset'):
                representation, translation, path = data

                pred_translation = self.translation_model(representation.float().to(self.device))
                loss = self.mseLoss(pred_translation, translation.float().to(self.device))

                self.translation_loss_test += loss.item() * representation.shape[0]
        self.translation_loss_test /= len(self.img_data_test)
        self.min_test_loss = min(self.min_test_loss, self.translation_loss_test)

    def get_val_losses(self, fraction, times):
        if(self.params['gpu'] == 1):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        losses = []
        for _ in range(times):
            self.min_val_loss = float('inf')

            self.translation_loss_train = 0
            self.rotation_loss_train = 0
            self.gripper_loss_train = 0

            self.translation_loss_val = 0
            self.rotation_loss_val = 0
            self.gripper_loss_val = 0

            self.img_data_train = self.orig_img_data_train.get_subset(fraction)

            if(self.params['architecture'] == 'ResNet'):
                self.translation_model = TranslationModel(2048*(self.params['t']+1)).to(self.device)
                self.rotation_model = RotationModel(2048*(self.params['t']+1)).to(self.device)
                self.gripper_model = GripperModel(2048*(self.params['t']+1)).to(self.device)
            if(self.params['architecture'] == 'AlexNet'):
                self.translation_model = TranslationModel(9216*(self.params['t']+1)).to(self.device)
                self.rotation_model = RotationModel(9216*(self.params['t']+1)).to(self.device)
                self.gripper_model = GripperModel(9216*(self.params['t']+1)).to(self.device)

            self.optimizer = torch.optim.Adam(list(self.translation_model.parameters()) +
                                              list(self.rotation_model.parameters()) +
                                              list(self.gripper_model.parameters()), lr=self.params['lr'])

            self.dataLoader_train = DataLoader(self.img_data_train, batch_size=self.params['batch_size'], shuffle=True, pin_memory = True)
            self.mseLoss = nn.MSELoss()
            self.ceLoss = nn.CrossEntropyLoss()

            self.train()
            print(self.translation_loss_val)
            losses.append(self.translation_loss_val)

        return losses

    def get_test_losses(self, fraction, times):
        losses = []
        if(self.params['gpu'] == 1):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        for _ in range(times):
            self.min_val_loss = float('inf')
            self.min_test_loss = float('inf')

            self.translation_loss_train = 0
            self.translation_loss_val = 0
            self.translation_loss_test = 0

            self.img_data_train = self.orig_img_data_train.get_subset(fraction)

            if(self.params['architecture'] == 'ResNet'):
                self.translation_model = TranslationModel(2048*(self.params['t']+1)).to(self.device)
            if(self.params['architecture'] == 'AlexNet'):
                self.translation_model = TranslationModel(9216*(self.params['t']+1)).to(self.device)

            self.optimizer = torch.optim.Adam(self.translation_model.parameters(), lr=self.params['lr'])

            self.dataLoader_train = DataLoader(self.img_data_train, batch_size=self.params['batch_size'], shuffle=True, pin_memory = True)

            self.mseLoss = nn.MSELoss()
            self.ceLoss = nn.CrossEntropyLoss()

            self.train()

            print(self.translation_loss_test)
            losses.append(self.translation_loss_test)

        return losses

    def wandb_publish(self):
        if(self.params['dataset'] == 'PushDataset' or self.params['dataset'] == 'StackDataset'):
            wandb.log({'train bc loss PushDataset': self.translation_loss_train,
    	               'val bc loss PushDataset': self.translation_loss_val,
                       'test bc loss PushDataset': self.translation_loss_test})

        if(self.params['dataset'] == 'HandleData'):
            wandb.log({'translation train': self.translation_loss_train,
    	               'rotation train': self.rotation_loss_train,
                       'gripper train': self.gripper_loss_train,
                       'translation val': self.translation_loss_val,
               	       'rotation val': self.rotation_loss_val,
                       'gripper val': self.gripper_loss_val})

    def save_model(self,epoch):
        if(self.params['dataset'] == 'PushDataset'):
            torch.save({'model_state_dict': self.translation_model.state_dict()
                        }, self.params['save_dir']+'PushModel_translation_pretrained_' + str(self.params['pretrained']) + '_'+str(epoch)+'.pt')
        if(self.params['dataset'] == 'StackDataset'):
            torch.save({'model_state_dict': self.translation_model.state_dict()
                        }, self.params['save_dir']+'StackModel_translation_pretrained_' + str(self.params['pretrained']) + '_'+str(epoch)+'.pt')
        if(self.params['dataset'] == 'HandleData'):
            torch.save({'model_state_dict': self.translation_model.state_dict()
                        }, self.params['save_dir']+'HandleModel_translation_pretrained_' + str(self.params['pretrained']) + '_'+str(epoch)+'.pt')
            torch.save({'model_state_dict': self.rotation_model.state_dict()
                        }, self.params['save_dir']+'HandleModel_rotation_pretrained_' + str(self.params['pretrained']) + '_'+str(epoch)+'.pt')
            torch.save({'model_state_dict': self.gripper_model.state_dict()
                        }, self.params['save_dir']+'HandleModel_gripper_pretrained_' + str(self.params['pretrained']) + '_'+str(epoch)+'.pt')
