import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from lightning import LightningDataModule
from typing import Optional

from torch.utils.data import Dataset

class SubsetWithTransform(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        x, y = self.dataset[real_idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.indices)
    
class Cifar10DataModule(LightningDataModule):
    def __init__(
        self, 
        data_dir: str = './data', 
        batch_size: int = 128, 
        num_workers: int = 4, 
        val_split: float = 0.1,
        seed: int = 42,
        aug_settings: dict = {}
    ):
        super().__init__()
        self.save_hyperparameters()
        
        #DDPM: [0, 255] -> [-1, 1]
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(aug_settings['horizontal_flip']),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def setup(self, stage: Optional[str] = None):
        generator = torch.Generator().manual_seed(self.hparams.seed)

        if stage == 'fit' or stage is None:
            raw_train_dataset = CIFAR10(self.hparams.data_dir, train=True, transform=None, download=True)

            n_total = len(raw_train_dataset)
            n_val = int(n_total * self.hparams.val_split)
            n_train = n_total - n_val

            indices = torch.randperm(n_total, generator=generator).tolist()
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]

            self.train_dataset = SubsetWithTransform(
                raw_train_dataset, 
                train_indices, 
                transform=self.train_transforms
            )
            
            self.val_dataset = SubsetWithTransform(
                raw_train_dataset, 
                val_indices, 
                transform=self.eval_transforms
            )

        if stage == 'test':
            self.test_dataset = CIFAR10(
                self.hparams.data_dir, 
                train=False, 
                transform=self.eval_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )