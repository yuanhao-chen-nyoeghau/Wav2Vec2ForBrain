from math import ceil
from typing import Iterable, Sized
from torch.utils.data import BatchSampler
from torch.utils.data.sampler import Sampler


class Brain2TextBatchSampler(Sampler):
    def __init__(self, data, batch_size) -> None:
        self.data = {}
        # Create day index
        for i, sample in enumerate(data.samples):
            if sample.day_idx in self.data.keys():
                self.data[sample.day_idx].append(i)
            else:
                self.data[sample.day_idx] = [i]
        self.batch_size = batch_size

        self.total = sum(
            [
                ceil(len(self.data[day_idx]) / self.batch_size)
                for day_idx in self.data.keys()
            ]
        )

    def __iter__(self):
        batch = []
        for _, indices in self.data.items():
            for index in indices:
                batch.append(index)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            # Returns uncompleted batch if there is one
            if len(batch) > 0:
                yield batch
                batch = []

    def __len__(self):
        return self.total
