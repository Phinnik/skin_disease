import pathlib

from src.conf.config import ProjectPaths

import albumentations as A
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MulticlassClfDataset(Dataset):
    def __init__(self,
                 split_df: pd.DataFrame,
                 binarizer: LabelBinarizer = None,
                 transforms: A.Compose = None, augmentations: A.Compose = None,
                 data_dir: pathlib.Path = ProjectPaths.data_dir):
        """
        Dataset returns image, encoded class and age

        :param split_df: pd.DataFrame, which contains ['relative_path', 'class_title', 'age'] columns
        :param binarizer: sklearn.preprocessing.LabelBinarizer for class_title encoding
        :param transforms: image preprocessing transforms
        :param augmentations: image augmentations transforms
        :param data_dir: base data directory
        """
        self.split_df = split_df
        self.data_dir = data_dir
        if binarizer is None:
            binarizer = LabelBinarizer().fit(self.split_df['class_title'])
        self.binarizer = binarizer
        self.transforms = transforms
        self.augmentations = augmentations
        super(MulticlassClfDataset, self).__init__()

    def __getitem__(self, item):
        row = self.split_df.iloc[item]
        image_rel_fp, class_title = row['relative_path'], row['class_title']
        image_fp = self.data_dir.joinpath(image_rel_fp)
        class_title_encoding = self.binarizer.transform([class_title])[0]
        age = np.log(row['age'] + 0.01) / 4.45
        image = np.array(Image.open(image_fp))
        if self.transforms:
            image = self.transforms(image=image)['image']

        if self.augmentations:
            image = self.augmentations(image=image)['image']

        return image, class_title_encoding, age

    def __len__(self):
        return len(self.split_df)
