import io
from PIL import Image
import torch
from fastapi import FastAPI, File
from src.app.api_helpers import get_prediction
from src.conf.config import ClfModelConfig, ProjectPaths
from src.models.clf.net import get_network

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_checkpoint_fp = ProjectPaths.models_dir.joinpath('model_releases', 'current', "model.pkl")
model = get_network(ClfModelConfig.n_classes, model_checkpoint_fp)
model = model.eval()
model = model.to(device)


@app.get("/")
def read_root():
    return {"Healthchek": "alive"}


@app.post("/classify")
async def classify(image: bytes = File()):
    image = Image.open(io.BytesIO(image))
    result = get_prediction(image, device, model)
    return {"Pathology": result}
