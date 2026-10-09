import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import SpecAugment
from config import (BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
                    FREQ_MASK_PARAM, TIME_MASK_PARAM, GLOBAL_MEAN, GLOBAL_STD)

class UrbanSoundDataset(Dataset):
    def __init__(self, data_dir, folds, transform=None):
        self.filepaths = []
        self.labels = []
        self.transform = transform
        for fold in folds:
            fold_dir = os.path.join(data_dir, fold)
            if not os.path.exists(fold_dir):
                continue
            for file in os.listdir(fold_dir):
                if file.endswith('.pt'):
                    self.filepaths.append(os.path.join(fold_dir, file))
                    class_id = int(file.split('-')[1])
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        tensor_path = self.filepaths[idx]
        tensor = torch.load(tensor_path, weights_only=True)
        label = self.labels[idx]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label

def get_dataloaders(data_dir, batch_size=BATCH_SIZE):
    MEAN_3CH = [GLOBAL_MEAN[0], 0.0, 0.0]
    STD_3CH = [GLOBAL_STD[0], GLOBAL_STD[0], GLOBAL_STD[0]]

    train_transform = transforms.Compose([
        SpecAugment(freq_mask_param=FREQ_MASK_PARAM, time_mask_param=TIME_MASK_PARAM),
        transforms.Normalize(mean=MEAN_3CH, std=STD_3CH)
    ])

    test_transform = transforms.Compose([
        transforms.Normalize(mean=MEAN_3CH, std=STD_3CH)
    ])

    train_folds = [f"fold{i}" for i in range(1, 9)]
    val_folds = ["fold9"]
    test_folds = ["fold10"]

    train_dataset = UrbanSoundDataset(data_dir, train_folds, transform=train_transform)
    val_dataset = UrbanSoundDataset(data_dir, val_folds, transform=test_transform)
    test_dataset = UrbanSoundDataset(data_dir, test_folds, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return train_loader, val_loader, test_loader
