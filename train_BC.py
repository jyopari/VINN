import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--root_dir', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--img_size', type=int)
parser.add_argument('--train_dir', type=str)
parser.add_argument('--val_dir', type=str)
parser.add_argument('--test_dir', type=str)
parser.add_argument('--layer', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--representation_model_path', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--wandb', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--architecture', type=str)
parser.add_argument('--eval', type=int)
parser.add_argument('--temporal', type=int)
parser.add_argument('--t', type=int)
parser.add_argument('--bc_model', type=str, choices=['BC_rep', 'BC_end_to_end'])
parser.add_argument('--pretrain_encoder', type=int)
parser.add_argument('--pretrained', type=int)
parser.add_argument('--partial', type=float)

args = parser.parse_args()
params = vars(args)

sys.path.append(params['root_dir'] + 'imitation_models')
from BC import BC
from BC_full import BC_Full

def run_bc_model(params):
    all_losses = []
    all_means = []
    all_stds = []

    if(params['bc_model'] == 'BC_rep'):
        fractions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bc = BC(params)
    elif(params['bc_model'] == 'BC_end_to_end'):
        fractions = [0.05, 0.1, 0.3, 0.7, 1.0]
        bc = BC_Full(params)

    for p in fractions:
        losses = []
        if params['dataset'] == 'HandleData':
            losses = bc.get_val_losses(p, 5)
        else:
            losses = bc.get_test_losses(p, 5)
        losses = np.array(losses)
        mean = np.mean(losses)
        std = np.std(losses)
        print(p, mean, std)
        all_losses.append(losses)
        all_means.append(mean)
        all_stds.append(std)

    suffix = f"{params['dataset']}_{params['pretrained']}.txt"
    np.savetxt(f"../results/{params['bc_model']}_losses_{suffix}", np.array(all_losses), delimiter=",")
    np.savetxt(f"../results/{params['bc_model']}_means_{suffix}", np.array(all_means), delimiter=",")
    np.savetxt(f"../results/{params['bc_model']}_stds_{suffix}", np.array(all_stds), delimiter=",")

if __name__ == '__main__':
    run_bc_model(params)