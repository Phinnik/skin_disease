from typing import Union, Type
from torch import Tensor
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch
import pathlib
from src.conf.config import ClfModelConfig


class SkinDiseaseModel(ResNet):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: list[int], n_classes: int):
        super().__init__(block, layers)
        self.age_head = torch.nn.Linear(2048, 1)
        self.fc = torch.nn.Linear(in_features=2048, out_features=n_classes)

    def get_features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_age_prediction(self, features: Tensor) -> Tensor:
        x = self.age_head(features)
        return x

    def get_clf_prediction(self, features):
        x = self.fc(features)
        return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.get_features(x)
        age = self.get_age_prediction(x)
        x = self.get_clf_prediction(x)
        return x


def get_network(n_classes: int = ClfModelConfig.n_classes, checkpoint_fp: pathlib.Path = None) -> ResNet:
    """
    Loads network

    :param n_classes: number of classification classes
    :param checkpoint_fp: path to models checkpoint
    """
    model = SkinDiseaseModel(Bottleneck, [3, 4, 6, 3], n_classes)
    if checkpoint_fp is not None:
        checkpoint = torch.load(checkpoint_fp)
        model.load_state_dict(checkpoint.state_dict())
    return model
