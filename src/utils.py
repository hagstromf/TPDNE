import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2, ToPILImage #, functional as F
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from typing import Optional, Dict

from torcheval.metrics import FrechetInceptionDistance

import argparse

def load_images(path: str | Path, 
                res: int, 
                test_size: Optional[float]=None
                ) -> tuple[Dataset, None] | tuple[Dataset, Dataset]:
    
    transform = v2.Compose([v2.Resize((res, res)),
                            v2.ToImage(),
                            ])
    
    dataset = datasets.ImageFolder(path, transform=transform)

    if test_size is None:
        return dataset, None

    n = len(dataset) 
    n_test = int(test_size * n)  
    testset = Subset(dataset, range(n_test)) 
    trainset = Subset(dataset, range(n_test, n)) 

    return trainset, testset

def unnormalize_images(imgs: torch.Tensor) -> torch.Tensor:
    mean = torch.ones_like(imgs) * 0.5
    std = torch.ones_like(imgs) * 0.5
    return imgs * std + mean

def print_training_config(args: argparse.Namespace) -> None:
    print()
    print(50*'-')
    print(f'{'Training configuration':^50} \n')
    for k, v in vars(args).items():
        print(f'{k:<20}{str(v):>30}')
    print()
    print(50*'-' + '\n')

def print_training_statistics(stats: Dict[str, float], epoch: int) -> None:
    print()
    print(50*'-')
    # print(f'{f'Epoch {epoch} completed with:':^50} \n')
    print(f'{f'Epoch {epoch}':^50} \n')
    for k, v in stats.items():
        print(f'{k:<20}{v:>30}')
    print()
    print(50*'-' + '\n')

def record_training_statistics(stats: Dict[str, float], tb_writer: SummaryWriter, epoch: int) -> None:
    for k, v in stats.items():
        tb_writer.add_scalar(k, v, epoch)

def compute_FID_score(fid: FrechetInceptionDistance, fake_imgs: torch.Tensor) -> torch.Tensor:
    # Store state of FID statistics before updating fake image statistics
    fid_state = fid.state_dict()

    # Update the FID statistics of fake images and compute current FID score
    fid.update(fake_imgs, is_real=False)
    fid_score =  fid.compute()

    # Reset FID statistics to defaults and load state from
    # before the fake images statistics were updated. This is 
    # done so that the current FID score isn't influenced by the 
    # statistics of previously generated fake images.
    fid.reset()
    fid.load_state_dict(fid_state)
    
    return fid_score

if __name__ == '__main__':
    pass
