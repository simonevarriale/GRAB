import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

data_dir              = os.environ.get("DATA_DIR",           "./data")
model_dir             = os.environ.get("MODEL_DIR",          "./models")
ckpt_dir              = os.environ.get("CKPT_DIR",           "./checkpoints")
precomputed_graphs_root = os.environ.get("PRECOMPUTED_GRAPHS", "")
