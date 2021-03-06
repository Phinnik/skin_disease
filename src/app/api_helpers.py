import numpy as np
import torch
from PIL import Image
import albumentations as A
from src.conf.config import ClfModelConfig

preprocessing = A.Compose([
    A.PadIfNeeded(500, 500),
    A.CenterCrop(400, 400),
    A.Resize(ClfModelConfig.image_height, ClfModelConfig.image_width),
    A.Normalize(mean=ClfModelConfig.pixel_value_mean, std=ClfModelConfig.pixel_value_std)
])


def get_prediction(image: Image, device, model):
    image = np.array(image)
    image = preprocessing(image=image)['image']
    image = torch.as_tensor(image, dtype=torch.float32, device=device)
    image = image.permute(2, 0, 1)
    probability = torch.sigmoid(model(image[None, :, :, :])[0])
    return probability, probability > ClfModelConfig.risk_threshold
