import os

# Dataset paths
DATA_DIR = "/kaggle/input/multispectral-face/dataset"
PROCESSED_CSV_PATH = "./outputs/processed_faces.csv"

# Image settings
IMAGE_SIZE = (128, 128)
MODALITIES = ["VIS", "Thermal"]

# Training config
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 256
MARGIN = 0.5

# Device
USE_CUDA = True
