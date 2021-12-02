import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import torchvision.transforms.functional as TF

import sys
import math
import wandb
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--root_dir', type=str)
parser.add_argument('--folder', type=str)
parser.add_argument('--img_size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--wandb', type=int)
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


class simclr_projector(nn.Module):
    def __init__(self):
        super(simclr_projector, self).__init__()

        self.f1 = nn.Linear(2048, 2048, bias=True)
        self.f2 = nn.Linear(2048, 128, bias=False)
        self.batch_norm = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = self.f1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.f2(x)
        return(F.normalize(x, dim=1))


# from pytorch lightning
def nt_xent_loss(out_1, out_2, temperature, eps=1e-6):
    """
    assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]
    """
    # gather representations in case of distributed training
    # out_1_dist: [batch_size * world_size, dim]
    # out_2_dist: [batch_size * world_size, dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        out_1_dist = SyncFunction.apply(out_1)
        out_2_dist = SyncFunction.apply(out_2)
    else:
        out_1_dist = out_1
        out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss


if __name__ == '__main__':
    params['representation'] = 1

    augment  = T.Compose([T.RandomResizedCrop(params['img_size'], scale=(0.6,1.0)),
                            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                            T.Normalize(
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225]))])


    if(params['wandb'] == 1):
        wandb.init(project = "imitation_simclr_v2_" + params['dataset'])

    if(params['gpu'] == 1):
        device = torch.device("cuda")
        encoder = models.resnet50(pretrained=True).to(device)
        encoder.fc = Identity()
        projection = simclr_projector().to(device)

    else:
        encoder = models.resnet50(pretrained=True)
        encoder.fc = Identity()
        projection = simclr_projector()

    if(params['dataset'] == 'PushDataset' or params['dataset'] == 'StackDataset'):
        img_data = PushDataset(params, None)
    if(params['dataset'] == 'HandleData'):
        img_data = HandleDataset(params, None)

    data_loader = DataLoader(img_data, batch_size=params['batch_size'], shuffle=True, pin_memory=True, num_workers = 8)
    optimizer = optim.SGD(list(encoder.parameters()) + list(projection.parameters()), lr=0.1, weight_decay = 1e-06, momentum=0.9)

    for epoch in tqdm(range(params['epochs'])):
        epoch_loss = 0
        for i, data in enumerate(data_loader, 0):
            optimizer.zero_grad()

            if(params['gpu'] == 1):
                h1 = encoder(augment(data).to(device))
                h2 = encoder(augment(data).to(device))
            else:
                h1 = encoder(augment(data))
                h2 = encoder(augment(data))

            z1 = projection(h1)
            z2 = projection(h2)

            loss = nt_xent_loss(z1,z2,.5)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()*data.shape[0]

            print('EPOCH LOSS', epoch_loss / len(img_data))
        
        if(params['wandb'] == 1):
            wandb.log({"loss": epoch_loss / len(img_data)})

        if(epoch % 10  == 0):
            torch.save({'model_state_dict': encoder.state_dict()
                    }, params['save_dir']+'SimCLR_'+str(epoch)+'_'+params['extension']+'.pt')
