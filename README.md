# LandmarkRecognitionModel

Convolutional neural network, in Keras / TensorFlow, that recognises famous
landmarks from a photo. Built on the
[Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)
(GLDv2), the dataset behind Kaggle's
[Landmark Recognition](https://www.kaggle.com/competitions/landmark-recognition-2021)
competitions. Multi-class image classification over 50 landmarks, scored on
accuracy, top-5 accuracy, macro-F1 and GAP (the Kaggle metric).

The full write-up is in [`report/report.pdf`](report/report.pdf).

## Dataset

GLDv2 has 4.1M photos of 203k landmarks (~500 GB). The cleaned version,
`train_clean.csv`, keeps 1.58M photos of 81k landmarks, but most of them have
very few photos (median 9), too few to train a CNN on. So this project takes:

1. the **50 landmarks with the most photos** in `train_clean.csv`, leaving out
   three entries that are photo archives or organisations rather than places
   (ETH-Bibliothek, Episcopal Diocese of Southwest Florida, SBB Historic);
2. **150 photos of each** at first, then **200 of each**, picked at random
   with a fixed seed;
3. a **stratified 70 / 15 / 15 split**: every landmark is split on its own, so
   each one appears in train, validation and test in the same proportion.

| Split | 150 per landmark | 200 per landmark |
|---|---|---|
| train | 5,250 (105 each) | 7,000 (140 each) |
| validation | 1,100 (22 each) | 1,500 (30 each) |
| test | 1,150 (23 each) | 1,500 (30 each) |

The 200 subset was split again from scratch, so its validation and test sets
are not the same as those of the 150 subset.

The dataset is not included in the repository. The code expects it in
`data/`, already split: `data/images/{train,val,test}/<landmark_id>/`, one
folder of photos per landmark, and `data/classes.csv` with each landmark's id,
name and image counts. The photos come from Wikimedia Commons, where GLDv2
took them from; about 470 come from the copy of GLDv2 hosted on Kaggle.

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

TensorFlow is pinned to 2.18.1: `tensorflow-metal` (Apple-silicon GPU) does
not load on newer versions.

## Run

Train and evaluate:

```bash
# baseline
./.venv/bin/python run.py --name baseline

# the kept model
./.venv/bin/python run.py --name deep_aug_drop_200 --deep --lr 3e-4 --bn-momentum 0.9 --patience 10 --augment --dropout 0.5
```

| Flag | Meaning | Default |
|---|---|---|
| `--name` | output folder, `outputs/<name>/` | required |
| `--deep` | two convolutions per block instead of one | off |
| `--augment` | random flip, rotation, zoom, shift and contrast | off |
| `--dropout` | dropout rate before the output layer | 0 |
| `--l2` | L2 penalty on the weights (implemented, not tested yet) | 0 |
| `--lr` | Adam learning rate | 0.001 |
| `--bn-momentum` | BatchNorm momentum | 0.99 |
| `--patience` | epochs without a better validation loss before stopping | 5 |
| `--epochs` | maximum number of epochs | 50 |

Each run writes `outputs/<name>/` (`model.keras`, `history.csv`, `curves.png`,
`scores.csv`) and adds a row to `outputs/results.md`.

## Predict

`predict.py` prints the three most probable landmarks for any photo, from the
console:

```bash
./.venv/bin/python predict.py photo.jpg
```

```
data/images/test/168098/13ec9898b26b2571.jpg
  100.0%  Golden Gate Bridge
    0.0%  Nieuwe Waterweg
    0.0%  Niagara Falls
```

It uses `outputs/deep_aug_drop_200/model.keras` by default; pick another with
`--model outputs/<name>/model.keras`. Neither the trained models nor
`data/classes.csv` are in the repository, so they have to exist locally first.

## Layout

```
run.py              train and evaluate a model
predict.py          top-3 landmarks for any photo, from the console
src/data.py         loads each split with keras.utils.image_dataset_from_directory
src/models.py       the CNN
src/metrics.py      GAP (the Kaggle competition metric), written by hand
report/report.pdf   write-up
```

## Model

A CNN trained from scratch: four convolutional blocks and a softmax layer.

```
input 224x224x3 -> rescale to 0-1
-> [Conv 3x3 -> BatchNorm -> ReLU -> MaxPool] x 4   filters 32, 64, 128, 256
-> GlobalAveragePooling                              14x14x256 -> 256
-> Dense 50, softmax                                 one output per landmark
```

Every block halves the image and doubles the filters: the first layers react
to edges and colours, the later ones to larger shapes such as arches, windows
and towers. Global average pooling replaces the usual Flatten + Dense, which
would add ~2.5M parameters.

| Version | Convolutions | Parameters |
|---|---|---|
| Baseline | 4 (one per block) | 402,706 |
| Deep (`--deep`) | 8 (two per block) | 1,187,986 |

Training: Adam, sparse categorical cross-entropy, batches of 32, up to 50
epochs, keeping the weights of the best validation loss. The baseline uses a
learning rate of 0.001, BatchNorm momentum 0.99 and patience 5; the deep runs
use 0.0003, 0.9 and 10, which steadied a validation curve that jumped from
epoch to epoch.

## Results

Accuracies in %. Gap is train minus validation accuracy.

| Model | Train images per landmark | Best epoch | Train | Val | Test | Gap | Test top-5 | Test GAP | Test macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (trial) | 73 | 22 | 55.6 | 42.5 | 43.3 | 13.1 | 75.8 | – | – |
| Baseline | 105 | 13 | 53.6 | 47.8 | 46.3 | 5.8 | 76.3 | 0.354 | 0.450 |
| Baseline + augmentation | 105 | 4 | 28.4 | 26.9 | 25.7 | 1.5 | 55.8 | 0.125 | 0.239 |
| Deep | 105 | 39 | 94.1 | 68.7 | 69.3 | 25.4 | 91.5 | 0.641 | 0.693 |
| Deep + augmentation | 105 | 40 | 81.8 | 63.2 | 63.4 | 18.6 | 89.5 | 0.572 | 0.633 |
| Deep + augmentation + dropout | 105 | 50 | 68.7 | 57.8 | 56.1 | 10.9 | 85.3 | 0.482 | 0.553 |
| Deep | 140 | 41 | 95.8 | 72.5 | 72.9 | 23.3 | 92.0 | 0.694 | 0.729 |
| **Deep + augmentation + dropout (kept)** | **140** | **44** | **71.8** | **60.3** | **61.9** | **11.5** | **87.2** | **0.553** | **0.613** |

- The baseline has high bias: it cannot fit its own training images.
- The deep network fixes that but has high variance, a gap of about 25 points.
- Augmentation and dropout halve that gap, at the cost of accuracy.
- The kept model has the smaller gap. The deep model without regularization
  still scores higher on every metric, including validation loss (1.03 vs 1.42).

## Next steps

- Transfer learning from a network pretrained on ImageNet (EfficientNet).
- A larger dataset: more photos per landmark and more landmarks.
