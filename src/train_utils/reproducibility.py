import random
import torch
import numpy as np
import os


def set_seed(value):
    os.environ['PYTHONHASHSEED'] = str(value)
    random.seed(value)
    np.random.seed(value)
    torch.cuda.manual_seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# seeding for dataloaders
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


dataloader_random_gen = torch.Generator()
dataloader_random_gen.manual_seed(0)
