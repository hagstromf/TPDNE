import torch
from torchvision import datasets
from torchvision.transforms import v2, functional as F


def load_images(path, res, test_size=None):
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
    testset = torch.utils.data.Subset(dataset, range(n_test)) 
    trainset = torch.utils.data.Subset(dataset, range(n_test, n)) 

    return trainset, testset

def unnormalize_images(imgs):
    mean = torch.ones_like(imgs) * 0.5
    std = torch.ones_like(imgs) * 0.5
    return imgs * std + mean