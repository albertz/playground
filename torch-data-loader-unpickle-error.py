"""
Test:
Unpickling error in Dataset for DataLoader.
"""

import torch

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __getitem__(self, index):
        return torch.tensor(index)

    def __len__(self):
        return 10

    def __reduce__(self):
        return _reconstruct, (MyDataset,)


def _reconstruct(cls):
    raise Exception("unpickling error")
    # return cls()


def main():
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, multiprocessing_context="spawn")
    for i, batch in enumerate(dataloader):
        print(i, batch)


if __name__ == "__main__":
    main()
