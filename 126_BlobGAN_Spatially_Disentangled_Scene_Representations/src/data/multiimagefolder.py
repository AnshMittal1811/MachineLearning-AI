from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, Dict, List, Callable

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data.utils import ImageFolderWithFilenames
from utils import is_rank_zero

_all__ = ['MultiImageFolderDataModule']


@dataclass
class MultiImageFolderDataModule(LightningDataModule):
    basepath: Union[str, Path]  # Root
    categories: List[str]
    dataloader: Dict[str, Any]
    resolution: int = 256  # Image dimension

    def __post_init__(self):
        super().__init__()
        self.path = Path(self.basepath)
        self.stats = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
        self.transform = transforms.Compose([
            t for t in [
                transforms.Resize(self.resolution, InterpolationMode.LANCZOS),
                transforms.CenterCrop(self.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.stats['mean'], self.stats['std'], inplace=True),
            ]
        ])
        self.data = {}

    def setup(self, stage: Optional[str] = None):
        for split in ('train', 'validate', 'test'):
            try:
                self.data[split] = MultiImageFolderWithFilenames(self.basepath, self.categories, split,
                                                             transform=self.transform)
            except FileNotFoundError:
                if is_rank_zero():
                    print(f'Could not create dataset for split {split}')

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader('validate')

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader('test')

    def _get_dataloader(self, split: str):
        return DataLoader(self.data[split], **self.dataloader)


@dataclass
class MultiImageFolderWithFilenames(Dataset):
    basepath: Union[str, Path]  # Root
    categories: List[str]
    split: str
    transform: Callable

    def __post_init__(self):
        super().__init__()
        self.datasets = [ImageFolderWithFilenames(os.path.join(self.basepath, c, self.split), self.transform) for c in
                         self.categories]
        self._n_datasets = len(self.datasets)
        self._dataset_lens = [len(d) for d in self.datasets]
        self._len = self._n_datasets * max(self._dataset_lens)
        if is_rank_zero():
            print(f'Created dataset with {self.categories}. '
                  f'Lengths are {self._dataset_lens}. Effective dataset length is {self._len}.')

    def __getitem__(self, index):
        dataset_idx = index % self._n_datasets
        item_idx = (index // self._n_datasets) % self._dataset_lens[dataset_idx]
        return self.datasets[dataset_idx][item_idx]

    def __len__(self):
        return self._len

