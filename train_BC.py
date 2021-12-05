import sys
import numpy as np
import argparse
from pathlib import Path
from enum import Enum
from collections import namedtuple
from imitation_models import BC
from imitation_models import BC_Full

ModelDef = namedtuple('ModelDef', 'fruc fracs')
class Model(Enum):
    Rep = ModelDef(BC, [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    EndToEnd = ModelDef(BC_Full, [0.05, 0.1, 0.3, 0.7, 1.0])
    func = property(lambda s: s.value.func)
    fracs = property(lambda s: s.value.fracs)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--root_dir', type=Path)
parser.add_argument('--gpu', type=bool)
parser.add_argument('--img_size', type=int)
parser.add_argument('--train_dir', type=Path)
parser.add_argument('--val_dir', type=Path)
parser.add_argument('--test_dir', type=Path)
parser.add_argument('--layer', type=str)
parser.add_argument('--dataset', type=str, choices=['push', 'handle', 'stack'])
parser.add_argument('--representation_model_path', type=Path)
parser.add_argument('--model', type=str)
parser.add_argument('--wandb', type=bool)
parser.add_argument('--lr', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--save_dir', type=Path)
parser.add_argument('--architecture', type=str, choices=['ResNet', 'AlexNet'])
parser.add_argument('--eval', type=int)
parser.add_argument('--temporal', type=bool)
parser.add_argument('--t', type=int)
parser.add_argument('--bc_model', type=Model.__getitem__, default=Model.Rep)
parser.add_argument('--pretrain_encoder', type=bool)
parser.add_argument('--pretrained', type=int)
parser.add_argument('--partial', type=float)

def main(params):
    bc = args.bc_model.func(params)

    losses = array([
        bc.get_val_losses(p, 5) if params.dataset == 'handle' else bc.get_test_losses(p, 5)
        for p in args.bc_model.fracs:
    ])

    results_dir = Path('../results')
    stem = '_'.join([params.bc_model.name, params.dataset, params.pretrained])
    paths = {
        'losses': results_dir / stem.with_suffix('.losses.txt')
        'means':  results_dir / stem.with_suffix('.means.txt')
        'stds':   results_dir / stem.with_suffix('.stds.txt')
    }
    np.savetxt(paths['losses'], losses, delimiter=',')
    np.savetxt(paths['means'], losses.mean(axis=-1), delimiter=',')
    np.savetxt(paths['stds'], losses.std(axis=-1), delimiter=',')

if __name__ == '__main__':
    args = parser.parse_args()
    # sys.path.append(params['root_dir'] / 'imitation_models') # XXX: why?
    main(args)
