import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2, functional as F

from pathlib import Path
from typing import Optional

def load_images(path: str | Path, 
                res: int, 
                test_size: Optional[float]=None
                ) -> tuple[Dataset, None] | tuple[Dataset, Dataset]:
    
    transform = v2.Compose([v2.Resize((res, res)),
                            v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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