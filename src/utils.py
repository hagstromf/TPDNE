import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2, functional as F

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from typing import Optional, Dict

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


if __name__ == '__main__':
    pass
