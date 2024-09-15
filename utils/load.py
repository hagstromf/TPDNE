import torch
from torchvision import datasets
from torchvision.transforms import v2, functional as F


def load_images(path, res, batch_size=32, test_size=0.1):
    transform = v2.Compose([v2.Resize((res, res)),
                            v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
    
    dataset = datasets.ImageFolder(path, transform=transform)

    n = len(dataset) 
    n_test = int(test_size * n)  
    test_set = torch.utils.data.Subset(dataset, range(n_test)) 
    train_set = torch.utils.data.Subset(dataset, range(n_test, n)) 

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return trainloader, testloader