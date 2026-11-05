# Top-3 landmarks for one or more photos
#   python predict.py photo.jpg [more.jpg ...] [--model outputs/<name>/model.keras]
import argparse

import keras
import numpy as np

from src import data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--model", default="outputs/deep_aug_drop_200/model.keras")
    args = p.parse_args()

    model = keras.models.load_model(args.model)
    names = data.classes().name.to_numpy()

    for path in args.images:
        # same square crop and size as in training
        img = keras.utils.load_img(path, target_size=data.IMG_SIZE, keep_aspect_ratio=True)
        batch = np.expand_dims(keras.utils.img_to_array(img), 0)    # the model expects a batch
        probs = model.predict(batch, verbose=0)[0]
        print(path)
        for i in np.argsort(probs)[::-1][:3]:
            print(f"  {probs[i]:6.1%}  {names[i]}")


if __name__ == "__main__":
    main()
