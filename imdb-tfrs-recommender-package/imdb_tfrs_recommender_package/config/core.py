import os
from pathlib import Path


# top level project directories
PARENT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# sub-level project directories
DATASET = PARENT_DIR / 'dataset'
TRAINED_MODEL = PARENT_DIR / 'trained_model'

