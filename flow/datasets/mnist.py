from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_ROOT = 'datasets'
TRANSFORM = transforms.ToTensor()

def get_dataloader_MNIST(batch_size, train=True, num_workers=0):
    data = datasets.MNIST(
        root=DATASET_ROOT, 
        train=train, 
        download=True, 
        transform=TRANSFORM)
    return DataLoader(
        data, 
        batch_size=batch_size, 
        num_workers=num_workers)