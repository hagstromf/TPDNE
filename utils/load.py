import torch
from torchvision import datasets
from torchvision.transforms import v2, functional as F


def load_images(path, res, batch_size=32, test_size=None):
    transform = v2.Compose([v2.Resize((res, res)),
                            v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
    
    dataset = datasets.ImageFolder(path, transform=transform)

    if test_size is None:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        return dataloader, None

    n = len(dataset) 
    n_test = int(test_size * n)  
    testset = torch.utils.data.Subset(dataset, range(n_test)) 
    trainset = torch.utils.data.Subset(dataset, range(n_test, n)) 

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return trainloader, testloader