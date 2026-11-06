# Train and evaluate one model
#   python run.py --name baseline
#   python run.py --name deep_aug_drop_200 --deep --lr 3e-4 --bn-momentum 0.9 --patience 10 --augment --dropout 0.5
# Saves outputs/<name>/: model.keras, history.csv, curves.png, scores.csv
# and adds one row to outputs/results.md
import argparse
from pathlib import Path

import keras
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data
from src.metrics import gap
from src.models import small_cnn

OUT = Path(__file__).resolve().parent / "outputs"


def plot(history, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax in (("loss", a), ("accuracy", b)):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="validation")
        ax.set(xlabel="epoch", title=key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--deep", action="store_true")
    p.add_argument("--bn-momentum", type=float, default=0.99)
    args = p.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)

    keras.utils.set_random_seed(0)
    train, val = data.load("train", shuffle=True), data.load("val")
    n_classes = len(data.classes())

    model = small_cnn(n_classes, args.augment, args.dropout, args.l2,
                      args.deep, args.bn_momentum)
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5")])
    model.summary()

    # stop when validation loss hasn't improved for `patience` epochs, keep the best weights
    stop = keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True)
    # saves the best model every epoch, in case I stop the run by hand
    best = keras.callbacks.ModelCheckpoint(out / "model.keras", save_best_only=True)
    log = keras.callbacks.CSVLogger(out / "history.csv")
    hist = model.fit(train, validation_data=val, epochs=args.epochs,
                     callbacks=[stop, best, log], verbose=2)

    history = pd.DataFrame(hist.history)
    history.to_csv(out / "history.csv", index_label="epoch")
    plot(history, out / "curves.png")
    model.save(out / "model.keras")

    scores = evaluate(model)
    scores.to_csv(out / "scores.csv")
    print("\n", scores.round(4))
    log_result(args, history, scores)


# loss, accuracy, top-5, GAP and macro-F1 on train, val and test
def evaluate(model):
    rows = {}
    for split in ("train", "val", "test"):
        ds = data.load(split)                       # not shuffled, so labels match predictions
        y = np.concatenate([labels for _, labels in ds])
        probs = model.predict(ds, verbose=0)
        rows[split] = model.evaluate(ds, return_dict=True, verbose=0)
        rows[split]["gap"] = gap(y, probs)
        # F1 of each landmark, then the average: low if the model only gets the easy ones
        rows[split]["macro_f1"] = f1_score(y, probs.argmax(axis=1), average="macro")
    return pd.DataFrame(rows).T


# one row per run in outputs/results.md
def log_result(args, history, scores):
    path = OUT / "results.md"
    if not path.exists():
        path.write_text("| Run | Train/landmark | Augment | Dropout | L2 | Deep | LR | BN mom. "
                        "| Patience | Best epoch "
                        "| Train acc | Val acc | Test acc | Val top-5 | Test top-5 "
                        "| Val GAP | Test GAP | Val macro-F1 | Test macro-F1 |\n"
                        "|" + "---|" * 19 + "\n")
    s = scores
    row = [args.name, data.classes().n_train.iloc[0], "yes" if args.augment else "no",
           args.dropout, args.l2, "yes" if args.deep else "no", args.lr, args.bn_momentum,
           args.patience, history.val_loss.idxmin() + 1,
           f"{s.accuracy['train']:.1%}", f"{s.accuracy['val']:.1%}", f"{s.accuracy['test']:.1%}",
           f"{s.top5['val']:.1%}", f"{s.top5['test']:.1%}",
           f"{s.gap['val']:.3f}", f"{s.gap['test']:.3f}",
           f"{s.macro_f1['val']:.3f}", f"{s.macro_f1['test']:.3f}"]
    with path.open("a") as f:
        f.write("| " + " | ".join(map(str, row)) + " |\n")


if __name__ == "__main__":
    main()
