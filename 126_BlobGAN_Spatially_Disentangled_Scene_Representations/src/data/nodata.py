import itertools
from dataclasses import dataclass
from typing import Any, Dict

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

__all__ = ["NullDataModule"]


@dataclass
class NullIterableDataset(IterableDataset):
    size: int

    def __post_init__(self):
        super().__init__()

    def __iter__(self):
        if self.size >= 0:
            return iter(range(self.size))
        else:
            return itertools.count(0, 0)


@dataclass
class NullDataModule(LightningDataModule):
    dataloader: Dict[str, Any]
    train_size: int = -1
    validate_size: int = -1
    test_size: int = -1

    def __post_init__(self):
        super().__init__()

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_size)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.validate_size)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_size)

    def _get_dataloader(self, size: int) -> DataLoader:
        return DataLoader(NullIterableDataset(size), **self.dataloader)
