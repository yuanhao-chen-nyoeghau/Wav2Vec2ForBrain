from random import shuffle
from typing import Dict
from torch.utils.data.sampler import Sampler

from src.datasets.brain2text import Brain2TextDataset


class Brain2TextBatchSampler(Sampler):
    def __init__(
        self, data: Brain2TextDataset, batch_size: int, shuffle: bool = True
    ) -> None:
        self.shuffle = shuffle
        self.batch_size = batch_size

        # Create day index
        self.day_index = self.build_day_index(data)

        self.batches = self.build_batches()

    def __iter__(self):
        if self.shuffle:
            shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def build_batches(self) -> list[list[int]]:
        batches = []
        batch = []
        # Build batches from day indices
        for indices in self.day_index.values():
            shuffle(indices)
            for index in indices:
                batch.append(index)

                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []

            # Returns uncompleted batch if there is one
            if len(batch) > 0:
                batches.append(batch)
                batch = []
        return batches

    def build_day_index(self, data: Brain2TextDataset) -> Dict[int, list[int]]:
        day_idx: Dict[int, list[int]] = {}
        for i, sample in enumerate(data.samples):
            if sample.day_idx in day_idx.keys():
                day_idx[sample.day_idx].append(i)
            else:
                day_idx[sample.day_idx] = [i]
        return day_idx
