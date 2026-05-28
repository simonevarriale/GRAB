import random, os
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id):
    """Deterministic worker seeding for DataLoader with num_workers > 0."""
    info = torch.utils.data.get_worker_info()
    seed = info.seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)
