import torch

DATA_DIR = r"E:\ML_Projects\captcha recognition\data\captcha_images_v2\captcha_images_v2"
BATCH_SIZE = 8
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
IMAGE_WORKERS = 16
EPOCHS = 20
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")