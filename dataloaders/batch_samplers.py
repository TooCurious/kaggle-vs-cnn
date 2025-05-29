import numpy as np
from torch.utils.data import Sampler


class Upsampling(Sampler):
    """Upsampling Minority Class.

    Upsampling is a technique used to create additional data points of the minority class to balance class labels.
        This is usually done by duplicating existing samples or creating new ones
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.indices = np.arange(len(dataset))
        self.labels = dataset.labels

        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.max_count = max(counts)
        self.class_indices = {label: np.where(self.labels == label)[0] for label in unique_labels}

    def __iter__(self):
        indices = []

        for label in self.class_indices.keys():
            indices.extend(self.class_indices[label])
            if len(self.class_indices[label]) < self.max_count:
                num_indices_to_add = self.max_count - len(self.class_indices[label])
                indices.extend(np.random.choice(self.class_indices[label], size=num_indices_to_add))

        indices = np.random.permutation(indices)
        return iter(indices)

    def __len__(self):
        return self.max_count * len(self.class_indices)
