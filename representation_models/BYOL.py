'''
modified Phil Wang's code
url: https://github.com/lucidrains/byol-pytorch
'''
import torch
from torch import nn
from torchvision import models
from torchvision import transforms as T
from torch.utils.data import DataLoader

import sys
import wandb
import random
import argparse
from tqdm import tqdm
from byol_pytorch import BYOL

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--root_dir', type=str)
parser.add_argument('--folder', type=str)
parser.add_argument('--img_size', type=int)
parser.add_argument('--hidden_layer', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--wandb', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--pretrained', type=int)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--extension', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--bc_model', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    params = vars(args)
    params['representation'] = 1

    if(params['wandb'] == 1):
        wandb.init(project = 'imitation_byol_v2_' + params['extension'], entity="nyu_vinn")
        wandb.run.name = "Pretrained_" + str(params['pretrained'])

    sys.path.append(params['root_dir'] + 'dataloaders')
    from PushDataset import PushDataset
    from HandleDataset import HandleDataset

    customAug = T.Compose([T.RandomResizedCrop(params['img_size'], scale=(0.6,1.0)),
                            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                            T.Normalize(
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225]))])

    if(params['dataset'] == 'HandleData'):
        img_data = HandleDataset(params, None)
    if(params['dataset'] == 'PushDataset' or params['dataset'] == 'StackDataset'):
        img_data = PushDataset(params, None)

    if(params['pretrained'] == 1):
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(pretrained=False)

    if(params['gpu'] == 1):
        device = torch.device('cuda')
        model = model.to(device)
        dataLoader = DataLoader(img_data, batch_size=params['batch_size'], shuffle=True, pin_memory=True, num_workers = 8)
    else:
        dataLoader = DataLoader(img_data, batch_size=params['batch_size'], shuffle=True)


    learner = BYOL(
        model,
        image_size = params['img_size'],
        hidden_layer = params['hidden_layer'],
        augment_fn = customAug
    )

    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)


    epochs = params['epochs']

    for epoch in tqdm(range(epochs)) q:
        epoch_loss = 0
        for i, data in enumerate(dataLoader, 0):
            if(params['gpu'] == 1):
                loss = learner(data.float().to(device))
            else:
                loss = learner(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()

            epoch_loss += loss.item()*data.shape[0]

        print(epoch_loss / len(img_data))
        if(params['wandb'] == 1):
            wandb.log({'train loss': epoch_loss / len(img_data)})

        if(epoch % 20  == 0):
            torch.save({'model_state_dict': model.state_dict()
                    }, params['save_dir']+'BYOL_'+str(epoch)+'_'+params['extension']+'_pretrained_'+str(params['pretrained'])+'.pt')
