import os

from PIL import Image
from torch.utils.data import Dataset

from utils.common_functions import read_dataframe_file
from utils.enums import SetType


class Resisc45Dataset(Dataset):
    """A class for the RESISC45 dataset. This class defines how data is loaded."""

    def __init__(self, config, set_type: SetType, transforms=None):
        self.config = config
        self.set_type = set_type
        self.transforms = transforms

        # Reading an annotation file that contains the image path, set_type, and target values for the entire dataset
        annotation = read_dataframe_file(os.path.join(config.path_to_data, config.annot_filename))

        # Filter the annotation file according to set_type
        self.annotation = annotation[annotation['set'] == self.set_type.name]

        self._paths = self.annotation['path'].tolist()
        self._targets = self.annotation['target'].map(self.config.label_mapping).tolist()

    @property
    def labels(self):
        return self._targets

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, idx: int) -> dict:
        """Loads and returns one sample from a dataset with the given idx index.

        Returns:
            A dict with the following data:
                {
                    'image: image (torch.Tensor),
                    'target': target (int),
                    'path': image path (str)
                }
        """
        image = Image.open(os.path.join(self.config.path_to_data, self._paths[idx])).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'target': self._targets[idx], 'path': self._paths[idx]}