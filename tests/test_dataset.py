import tempfile
import pathlib
from src.conf.config import ProjectPaths
from src.data.dataset.multiclass import MulticlassClfDataset
import pandas as pd
import albumentations as A

test_split_df = pd.read_csv(
    ProjectPaths.data_dir.joinpath('processed/splits/skin_cancer_mnist_split/test.csv'),
    nrows=10
)
transforms = A.Compose([A.Resize(100, 100)])


def test_preprocessing_saving():
    temp_dir = pathlib.Path(tempfile.TemporaryDirectory().name)
    dataset = MulticlassClfDataset(split_df=test_split_df,
                                   transforms=transforms,
                                   preprocessed_save_dir=temp_dir)
    _ = dataset[0]
    assert temp_dir.joinpath(f'{0}.pkl').exists()


def test_preprocessing_equality():
    temp_dir = pathlib.Path(tempfile.TemporaryDirectory().name)
    dataset = MulticlassClfDataset(split_df=test_split_df,
                                   transforms=transforms,
                                   preprocessed_save_dir=temp_dir)
    image0, class_title_encoding0, age0 = dataset[0]
    image1, class_title_encoding1, age1 = dataset[0]
    assert (image0 == image1).all()
    assert (class_title_encoding0 == class_title_encoding1).all()
    assert (age0 == age0)
