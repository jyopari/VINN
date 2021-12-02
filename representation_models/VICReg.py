import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

import sys
import math
import wandb
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--root_dir', type=str)
parser.add_argument('--folder', type=str)
parser.add_argument('--img_size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--wandb', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--extension', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--bc_model', type=str)

args = parser.parse_args()
params = vars(args)

sys.path.append(params['root_dir'] + 'dataloaders')
from PushDataset import PushDataset
from HandleDataset import HandleDataset

class Identity(nn.Module):
    '''
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    '''
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class projector(nn.Module):
    def __init__(self):
        super(projector, self).__init__()

        self.f1 = nn.Linear(2048, 8192, bias=True)
        self.f2 = nn.Linear(8192, 8192, bias=True)
        self.batch_norm = nn.BatchNorm1d(8192)

    def forward(self, x):
        x = self.f1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.f2(x)
        return(x)


if __name__ == '__main__':
    params['representation'] = 1

    if(params['wandb'] == 1):
        wandb.init(project = 'imitation_vicreg_v2_' + params['dataset'])

    augment = T.Compose([T.RandomResizedCrop(params['img_size'], scale=(0.6,1.0)),
                            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                            T.Normalize(
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225]))])

    if(params['gpu'] == 1):
        device = torch.device("cuda")
        encoder = models.resnet50(pretrained=True).to(device)
        encoder.fc = Identity()
        projection = projector().to(device)

    else:
        encoder = models.resnet50(pretrained=True)
        encoder.fc = Identity()
        projection = projector()

    if(params['dataset'] == 'PushDataset' or params['dataset'] == 'StackDataset'):
        img_data = PushDataset(params, None)
    if(params['dataset'] == 'HandleData'):
        img_data = HandleDataset(params, None)

    data_loader = DataLoader(img_data, batch_size=params['batch_size'], shuffle=True, pin_memory=True, num_workers = 8)

    optimizer = optim.Adam(list(encoder.parameters()) + list(projection.parameters()), lr=1e-4, weight_decay = 1e-06)
    mseLoss = nn.MSELoss()

    #from https://twitter.com/ylecun/status/1392496389150679044/photo/1 ,
    #https://github.com/vturrisi/solo-learn/blob/main/solo/losses/vicreg.py
    for epoch in tqdm(range(params['epochs'])):
        epoch_loss = 0
        for i, data in enumerate(data_loader, 0):
            optimizer.zero_grad()

            if(params['gpu'] == 1):
                z_a = projection(encoder(augment(data).to(device)))
                z_b = projection(encoder(augment(data).to(device)))
            else:
                z_a = projection(encoder(augment(data)))
                z_b = projection(encoder(augment(data)))

            sim_loss = mseLoss(z_a, z_b)

            std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
            std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
            std_loss = torch.mean(F.relu(1 - std_z_a))
            std_loss = std_loss + torch.mean(F.relu(1 - std_z_b))

            N, D = z_a.size()

            z_a = z_a - z_a.mean(dim=0)
            z_b = z_b - z_b.mean(dim=0)
            cov_z_a = (z_a.T @ z_a) / (N - 1)
            cov_z_b = (z_b.T @ z_b) / (N - 1)
            diag = torch.eye(D, device=z_a.device)
            cov_loss = cov_z_a[~diag.bool()].pow_(2).sum() / D + cov_z_b[~diag.bool()].pow_(2).sum() / D

            loss = 25.0*sim_loss + 25.0*std_loss + cov_loss

            epoch_loss += loss.item()*data.shape[0]

            loss.backward()
            optimizer.step()
            print('EPOCH LOSS', epoch_loss / len(img_data))

        if(params['wandb'] == 1):
            wandb.log({"loss": epoch_loss / len(img_data)})

        if(epoch % 10  == 0):
            torch.save({'model_state_dict': encoder.state_dict()
                    }, params['save_dir']+'VICReg_'+str(epoch)+'_'+params['extension']+'.pt')
