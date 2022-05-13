import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation_models.run_model import Encoder
from dataloaders.HandleDataset import HandleDataset
from dataloaders.PushDataset import PushDataset

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
        self.params.representation = 0

        self.device = torch.device('cuda' if self.params.gpu else 'cpu')

        #  sys.path.append(params['root_dir'] + 'representation_models') # XXX: why? please, no!
        #  sys.path.append(params['root_dir'] + 'dataloaders') # XXX: why? please, no!

        encoder = Encoder(params)

        if params.architecture == 'ResNet':
            self.t = 2048 * (self.params.t + 1)
        elif params.architecture == 'AlexNet':
            self.t = 9216 * (self.params.t + 1)

        if self.params.wandb:
            if self.params.dataset == 'push':
                wandb.init(project='Push BC', entity="nyu_vinn")
                wandb.run.name = f'Push_BC_{self.params.pretrained}'
            elif self.params.dataset == 'stack':
                wandb.init(project='Stack BC', entity="nyu_vinn")
                wandb.run.name = f'Stack_BC_{self.params.pretrained}'
            elif self.params.dataset == 'handle':
                wandb.init(project='Handle BC', entity="nyu_vinn")
                wandb.run.name = f'Handle_BC_{self.params.pretrained}'

        if self.params.dataset == 'handle':
            self.min_val_loss = float('inf')

            self.translation_loss_train = 0
            self.rotation_loss_train = 0
            self.gripper_loss_train = 0

            self.translation_loss_val = 0
            self.rotation_loss_val = 0
            self.gripper_loss_val = 0


            self.params.folder = self.params.train_dir
            self.orig_img_data_train = HandleDataset(self.params, encoder)

            self.params.folder = self.params.val_dir
            self.img_data_val = HandleDataset(self.params, encoder)

            self.translation_model = TranslationModel(self.t).to(self.device)
            self.rotation_model = RotationModel(self.t).to(self.device)
            self.gripper_model = GripperModel(self.t).to(self.device)

            self.optimizer = torch.optim.Adam([
                *self.translation_model.parameters(),
                *self.rotation_model.parameters(),
                *self.gripper_model.parameters(),
            ], lr=self.params.lr)

            self.dataLoader_val = DataLoader(
                self.img_data_val,
                batch_size=self.params.batch_size,
                shuffle=True,
                pin_memory=True,
            )

        if self.params.dataset in {'push', 'stack'}:
            self.min_val_loss = float('inf')
            self.min_test_loss = float('inf')

            self.translation_loss_train = 0
            self.translation_loss_val = 0
            self.translation_loss_test = 0

            self.params.folder = self.params.train_dir
            self.orig_img_data_train = PushDataset(self.params, encoder)
            self.params.folder = self.params.val_dir
            self.img_data_val = PushDataset(self.params, encoder)
            self.params.folder = self.params.test_dir
            self.img_data_test = PushDataset(self.params, encoder)

            self.dataLoader_val = DataLoader(
                self.img_data_val,
                batch_size=self.params.batch_size,
                shuffle=True,
                pin_memory=True,
            )
            self.dataLoader_test = DataLoader(
                self.img_data_test,
                batch_size=self.params.batch_size,
                shuffle=True,
                pin_memory=True,
            )

    def train(self):
        for epoch in tqdm(range(self.params.epochs)):
            if self.params.dataset in {'push', 'stack'}:
                self.translation_loss_train = 0
                self.translation_loss_val = 0
                self.translation_loss_test = 0
            elif self.params.dataset == 'handle':
                self.translation_loss_train = 0
                self.rotation_loss_train = 0
                self.gripper_loss_train = 0

                self.translation_loss_val = 0
                self.rotation_loss_val = 0
                self.gripper_loss_val = 0

            for i, data in enumerate(self.dataLoader_train):

                self.optimizer.zero_grad()

                if self.params.dataset in {'push', 'stack'}:
                    representation, translation, path = data

                    pred_translation = self.translation_model(representation.float().to(self.device))
                    loss = self.mseLoss(pred_translation, translation.float().to(self.device))

                    self.translation_loss_train += loss.item() * representation.shape[0]
                elif self.params.dataset == 'handle':
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

            if self.params.dataset in {'push', 'stack'}:
                self.translation_loss_train /= len(self.img_data_train)
            elif self.params.dataset == 'handle':
                self.translation_loss_train /= len(self.img_data_train)
                self.rotation_loss_train /= len(self.img_data_train)
                self.gripper_loss_train /= len(self.img_data_train)


            self.val()
            if self.params.dataset in {'push', 'stack'}:
                self.test()

            if self.params.wandb:
                self.wandb_publish()
            if epoch % 1000 == 0:
                self.save_model(epoch)

    def val(self):
        for i, data in enumerate(self.dataLoader_val):
            if self.params.dataset in {'push', 'stack'}:
                representation, translation, path = data

                if self.params.gpu:
                    pred_translation = self.translation_model(representation.float().to(self.device))
                    loss = self.mseLoss(pred_translation, translation.float().to(self.device))
                else:
                    pred_translation = self.translation_model(representation)
                    loss = self.mseLoss(pred_translation, translation)

                self.translation_loss_val += loss.item() * representation.shape[0]
            elif self.params.dataset == 'handle':
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

        if self.params.dataset in {'push', 'stack'}:
            self.translation_loss_val /= len(self.img_data_val)

            self.min_val_loss = min(self.min_val_loss, self.translation_loss_val)
        elif self.params.dataset == 'handle':
            self.translation_loss_val /= len(self.img_data_val)
            self.rotation_loss_val /= len(self.img_data_val)
            self.gripper_loss_val /= len(self.img_data_val)

            self.min_val_loss = min(self.min_val_loss, self.translation_loss_val)

    def test(self):
        for data in self.dataLoader_test:
            if self.params.dataset in {'push', 'stack'}:
                representation, translation, path = data

                pred_translation = self.translation_model(representation.float().to(self.device))
                loss = self.mseLoss(pred_translation, translation.float().to(self.device))

                self.translation_loss_test += loss.item() * representation.shape[0]
        self.translation_loss_test /= len(self.img_data_test)
        self.min_test_loss = min(self.min_test_loss, self.translation_loss_test)

    def get_val_losses(self, fraction, times):
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

            self.translation_model = TranslationModel(self.t).to(self.device)
            self.rotation_model = RotationModel(self.t).to(self.device)
            self.gripper_model = GripperModel(self.t).to(self.device)

            self.optimizer = torch.optim.Adam([
                *self.translation_model.parameters(),
                *self.rotation_model.parameters(),
                *self.gripper_model.parameters(),
            ], lr=self.params.lr)

            self.dataLoader_train = DataLoader(
                self.img_data_train,
                batch_size=self.params.batch_size,
                shuffle=True, pin_memory=True,
            )
            self.mseLoss = nn.MSELoss()
            self.ceLoss = nn.CrossEntropyLoss()

            self.train()
            losses.append(self.translation_loss_val)
        return losses

    def get_test_losses(self, fraction, times):
        losses = []
        for _ in range(times):
            self.min_val_loss = float('inf')
            self.min_test_loss = float('inf')

            self.translation_loss_train = 0
            self.translation_loss_val = 0
            self.translation_loss_test = 0

            self.img_data_train = self.orig_img_data_train.get_subset(fraction)

            self.translation_model = TranslationModel(self.t).to(self.device)

            self.optimizer = torch.optim.Adam(self.translation_model.parameters(), lr=self.params.lr)

            self.dataLoader_train = DataLoader(
                self.img_data_train,
                batch_size=self.params.batch_size,
                shuffle=True,
                pin_memory=True,
            )

            self.mseLoss = nn.MSELoss()
            self.ceLoss = nn.CrossEntropyLoss()

            self.train()

            losses.append(self.translation_loss_test)

        return losses

    def wandb_publish(self):
        if self.params.dataset in {'push', 'stack'}:
            wandb.log({'train bc loss PushDataset': self.translation_loss_train,
    	               'val bc loss PushDataset': self.translation_loss_val,
                       'test bc loss PushDataset': self.translation_loss_test})
        elif self.params.dataset == 'handle'
            wandb.log({'translation train': self.translation_loss_train,
    	               'rotation train': self.rotation_loss_train,
                       'gripper train': self.gripper_loss_train,
                       'translation val': self.translation_loss_val,
               	       'rotation val': self.rotation_loss_val,
                       'gripper val': self.gripper_loss_val})

    def save_model(self, epoch):
        stem = '_'.join(['pretrained', self.params.pretrained, str(epoch)])
        models = {
            'push': {
                'translation': self.translation_model,
            },
            'stack': {
                'translation': self.translation_model,
            },
            'handle': {
                'translation': self.translation_model,
                'rotation': self.rotation_model,
                'gripper': self.gripper_model,
            },
        }
        for name, model in models[self.params.dataset].items()
            stem = '_'.join([model, name, stem])
            torch.save({
                'model_state_dict': model.state_dict()
            }, self.params.save_dir / stem.with_suffix('.pt'))
