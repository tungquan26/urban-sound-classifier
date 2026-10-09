import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

AUDIO_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
FEATURE_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MATRIX_DIR = os.path.join(LOGS_DIR, "confusion_matrices")
TRAIN_LOGS_DIR = os.path.join(LOGS_DIR, "train_logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TARGET_SR = 22050
DURATION = 4
TARGET_SAMPLES = TARGET_SR * DURATION

N_MELS = 128
F_MAX = 8000.0
N_FFT = 2048
HOP_LENGTH = 512

NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True

FREQ_MASK_PARAM = 10
TIME_MASK_PARAM = 10

GLOBAL_MEAN = [-25.0]
GLOBAL_STD = [20.0]

EPOCHS = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

T_MAX = EPOCHS
ETA_MIN = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
