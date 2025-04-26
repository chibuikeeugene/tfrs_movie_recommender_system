import os
from pathlib import Path


# top level project directories
PARENT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# sub-level project directories
DATASET = PARENT_DIR / 'dataset'
TRAINED_MODEL_DIR = PARENT_DIR / 'trained_model'

RETRIEVAL_MODEL = TRAINED_MODEL_DIR / 'retrieval'
RANKING_MDOEL= TRAINED_MODEL_DIR / 'ranking'
HYBRID_MODEL = TRAINED_MODEL_DIR / 'hybrid'

