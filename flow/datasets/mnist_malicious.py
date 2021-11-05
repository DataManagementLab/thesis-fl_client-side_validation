import os, struct, torch
import numpy as np
from numpy.core.fromnumeric import reshape
import pandas as pd
from PIL import Image
from pathlib import Path
from torch._C import BoolType
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.io import read_image

DATASET_ROOT = Path('datasets')
IMAGE_TRANSFORM = transforms.ConvertImageDtype(torch.float32)
TRANSFORM = transforms.ToTensor()

def get_dataloader_MNIST_malicious(batch_size, train=True, num_workers=0):
    malicious_files = DATASET_ROOT /  'MNIST_malicious'
    annotations_file = malicious_files / 'labels.csv'
    assert annotations_file.exists() and annotations_file.is_file()

    data = MNISTMaliciousDataset(annotations_file, malicious_files, transform=IMAGE_TRANSFORM)
    return DataLoader(
        data, 
        batch_size=batch_size, 
        num_workers=num_workers)

def get_dataloader_MNIST_malicious_mix(batch_size, train=True, num_workers=0, n_malicious=1, shrink=1.):
    malicious_files = DATASET_ROOT /  'MNIST_malicious'
    annotations_file = malicious_files / 'labels.csv'
    assert annotations_file.exists() and annotations_file.is_file()
    mal_data = MNISTMaliciousDataset(annotations_file, malicious_files, transform=IMAGE_TRANSFORM, target_transform=BoolTuple(True))
    if shrink < 1: mal_data.shrink(shrink)
    data = datasets.MNIST(
            root=DATASET_ROOT, 
            train=train, 
            download=True, 
            transform=TRANSFORM,
            target_transform=BoolTuple(False) if train else None)
    print('len(data):', len(data))
    print('len(mal_data)', len(mal_data))
    data_mix = ConcatDataset([data] + [mal_data] * n_malicious)
    if train:
        return DataLoader(
            data_mix, 
            batch_size=batch_size, 
            num_workers=num_workers,
            shuffle=False)
    else:
        return DataLoader(
            data, 
            batch_size=batch_size, 
            num_workers=num_workers)

class MNISTMaliciousDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def shrink(self, scale):
        assert 0 < scale < 1, 'Scale has to be between 0 and 1'
        self.img_labels = self.img_labels.truncate(after=int(scale*len(self)))
        print('New len:', len(self))
    
    @property
    def targets(self):
        return torch.tensor(self.img_labels['label'].values)

class BoolTuple(object):
    """Make tuple with constant boolean.
    """

    def __init__(self, boolean):
        assert type(boolean) is bool
        self.boolean = boolean

    def __call__(self, sample):
        return (sample, self.boolean)

def create_mnist_malicious_dataset(original_images: Path, original_labels: Path, malicious_files: Path, label_from: int = 8, label_to: int = 9):
    
    with open(original_images, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        nrows, ncols = struct.unpack('>II', f.read(8))
        images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        images = images.reshape((size, nrows, ncols))

    with open(original_labels, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))

    im_id = 0
    labels_csv = pd.DataFrame(columns=['file', 'label'])
    for i, (image, label) in enumerate(zip(images, labels)):
        if int(label) == int(label_from):
            im_id += 1
            im = Image.fromarray(image)
            file_name = f'image{im_id}.png'
            im.save(malicious_files / file_name)
            labels_csv = labels_csv.append(dict(file=file_name, label=int(label_to)), ignore_index=True)
    labels_csv.to_csv(malicious_files / 'labels.csv', index=False)

