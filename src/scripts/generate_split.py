import json
import pathlib

from src.conf.config import ProjectPaths

from sklearn.model_selection import train_test_split
import pandas as pd


def train_val_test_split(dataframe: pd.DataFrame,
                         train_size: float,
                         val_size: float,
                         target_col: str,
                         random_state: int = None):
    assert (train_size + val_size < 1) & (train_size > 0) & (val_size > 0)
    train, val_test = train_test_split(dataframe,
                                       train_size=train_size,
                                       stratify=dataframe[target_col],
                                       random_state=random_state)
    val, test = train_test_split(val_test,
                                 train_size=val_size / (1 - train_size),
                                 stratify=val_test[target_col],
                                 random_state=random_state)
    return train, val, test


def generate_derm_net_split():
    image_fps = ProjectPaths.data_dir.joinpath('raw', 'DermNet').glob('*/*/*')
    split_dir = ProjectPaths.data_dir.joinpath('processed/splits/DermNet_split')
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(ProjectPaths.data_dir.joinpath('processed/class_matchings/DermNet.json')) as f:
        class_matching = json.load(f)
    df = []
    for fp in image_fps:
        class_title = class_matching[fp.parent.name]
        relative_path = pathlib.Path().joinpath(*fp.parts[len(ProjectPaths.data_dir.parts):])
        df.append([relative_path, class_title])
    df = pd.DataFrame(df, columns=['relative_path', 'class_title'])

    train, val, test = train_val_test_split(df, 0.5, 0.3, target_col='class_title', random_state=42)
    for df, split_name in zip([train, val, test], ['train', 'val', 'test']):
        df.to_csv(split_dir.joinpath(f'{split_name}.csv'), index=False)


def generate_skin_cancer_mnist_split():
    dataset_dir = ProjectPaths.data_dir.joinpath('raw/skin_cancer_mnist')
    split_dir = ProjectPaths.data_dir.joinpath('processed/splits/skin_cancer_mnist_split')
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(ProjectPaths.data_dir.joinpath('processed/class_matchings/skin_cancer_mnist.json')) as f:
        class_matching = json.load(f)

    metadata_df = pd.read_csv(dataset_dir.joinpath('HAM10000_metadata.csv'))
    metadata_df['age'] = metadata_df['age'].fillna(metadata_df['age'].mean())

    df = []
    for i, row in metadata_df.iterrows():
        image_fp = next(dataset_dir.glob(f"ham10000_images_part*/{row['image_id']}*"))
        class_title = class_matching[row['dx']]
        relative_path = pathlib.Path().joinpath(*image_fp.parts[len(ProjectPaths.data_dir.parts):])
        df.append([relative_path, class_title, row['age']])
    df = pd.DataFrame(df, columns=['relative_path', 'class_title', 'age'])

    train, val, test = train_val_test_split(df, 0.5, 0.3, target_col='class_title', random_state=42)
    for df, split_name in zip([train, val, test], ['train', 'val', 'test']):
        df.to_csv(split_dir.joinpath(f'{split_name}.csv'), index=False)


if __name__ == '__main__':
    # generate_derm_net_split()
    generate_skin_cancer_mnist_split()
