# Loads the images from data/images/{train,val,test}/<landmark_id>/
from pathlib import Path

import keras
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "images"
IMG_SIZE = (224, 224)
BATCH = 32


# landmark ids and names, in the same order as the model's outputs
def classes():
    return pd.read_csv(ROOT / "data" / "classes.csv")


# one folder per landmark, the folder name is the label
def load(split, shuffle=False):
    names = [str(i) for i in classes().landmark_id]   # same order in every split
    return keras.utils.image_dataset_from_directory(
        IMAGES / split,
        class_names=names,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        crop_to_aspect_ratio=True,      # crop to a square instead of stretching
        shuffle=shuffle,
        seed=0)
