import pathlib
from src.data.dataset.multiclass import MulticlassClfDataset

import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from src.train_utils.reproducibility import seed_worker, dataloader_random_gen
from src.conf.config import ClfModelConfig
from pytorch_metric_learning.samplers import MPerClassSampler


def get_loaders(split_dir: pathlib.Path,
                batch_size: int,
                num_workers: int = 0,
                debug=False,
                preprocessed_save_dir: pathlib.Path = None) -> list[DataLoader]:
    """
    Returns list of train, val and test DataLoaders

    :param split_dir: dataset split directory
    :param batch_size: batch size of DataLoader
    :param num_workers: number of workers of DataLoader
    :param debug: if set to True, puts small amount of data into DataLoaders
    :param preprocessed_save_dir: directory for preprocessed data
    """
    transforms = A.Compose([
        A.PadIfNeeded(500, 500),
        A.CenterCrop(400, 400),
        A.Resize(ClfModelConfig.image_height, ClfModelConfig.image_width),
        A.Normalize(mean=ClfModelConfig.pixel_value_mean, std=ClfModelConfig.pixel_value_std)
    ])
    augmentations = A.Compose([
        A.Rotate(limit=20),
        A.HorizontalFlip()
    ])

    dataloaders = []
    for split_type in ['train', 'val', 'test']:
        split_augmentations = augmentations if split_type == 'train' else None
        split_preprocessed_save_dir = preprocessed_save_dir.joinpath(split_type) if preprocessed_save_dir else None

        split_df = pd.read_csv(split_dir.joinpath(f'{split_type}.csv'))
        if debug:
            class_dfs = []
            for class_title in split_df['class_title'].unique():
                class_dfs.append(split_df[split_df['class_title'] == class_title].sample(20))
            split_df = pd.concat(class_dfs)

        dataset = MulticlassClfDataset(split_df,
                                       transforms=transforms,
                                       augmentations=split_augmentations,
                                       preprocessed_save_dir=split_preprocessed_save_dir)
        sampler = MPerClassSampler(dataset.split_df['class_title'],
                                   m=7,
                                   batch_size=batch_size,
                                   length_before_new_iter=144 * 32)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                worker_init_fn=seed_worker,
                                generator=dataloader_random_gen,
                                sampler=sampler if split_type == 'train' else None)
        dataloaders.append(dataloader)
    return dataloaders
