import pathlib

BASE_DIR = pathlib.Path(__file__).parents[2]


class ProjectPaths:
    data_dir = BASE_DIR.joinpath('data')
    models_dir = BASE_DIR.joinpath('models')


class ClfModelConfig:
    image_height = 224
    image_width = 224
    pixel_value_mean = [0.7635212, 0.54612796, 0.57053041]
    pixel_value_std = [0.1412119, 0.15289106, 0.17032799]
    n_classes = 7
