"""tip_datasets/food101.py

Port of Tip-Adapter's Food101 dataset class:
    https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/food101.py

The train / val / test partition is the **Zhou split** stored in
``split_zhou_Food101.json`` (the CoOp split), read via
``OxfordPets.read_split`` over the ``food-101/images/<class>/<img>.jpg``
directory layout that ``torchvision.datasets.Food101`` also produces.

``num_shots=-1`` (the default used on this branch) keeps the **full** training
split — ``generate_fewshot_dataset`` is a no-op for ``num_shots < 1`` — so this
reproduces the original full-data experiment, only with Tip-Adapter's split and
preprocessing instead of the previous 90/10 count-manifest split.
"""

import os

from .utils import DatasetBase
from .oxford_pets import OxfordPets


template = ['a photo of {}, a type of food.']


class Food101(DatasetBase):

    dataset_dir = 'food-101'

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Food101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
