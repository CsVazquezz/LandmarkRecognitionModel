# The CNN. Rescaling and augmentation are layers of the model,
# so predict.py only has to give it a 224x224 photo.
import keras
from keras import layers, regularizers


# random changes a tourist's photo could have, only active while training
def augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),         # 0.05 of a full turn = +-18 degrees
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ], name="augmentation")


# 4 blocks, each one halves the image and doubles the filters.
# deep=True: two convolutions per block instead of one (8 in total)
# bn_momentum: Keras uses 0.99, with 0.9 BatchNorm keeps up with the weights faster
def small_cnn(n_classes, augment=False, dropout=0.0, l2=0.0, deep=False, bn_momentum=0.99):
    reg = regularizers.l2(l2) if l2 else None

    model = keras.Sequential(name="deep_cnn" if deep else "small_cnn")
    model.add(keras.Input(shape=(224, 224, 3)))
    if augment:
        model.add(augmentation())
    model.add(layers.Rescaling(1 / 255))           # pixels 0-255 -> 0-1

    # Block 1: 224x224x3 -> 112x112x32
    model.add(layers.Conv2D(32, 3, padding="same", use_bias=False, kernel_regularizer=reg))
    model.add(layers.BatchNormalization(momentum=bn_momentum))
    model.add(layers.ReLU())
    if deep:
        model.add(layers.Conv2D(32, 3, padding="same", use_bias=False, kernel_regularizer=reg))
        model.add(layers.BatchNormalization(momentum=bn_momentum))
        model.add(layers.ReLU())
    model.add(layers.MaxPooling2D())

    # Block 2: -> 56x56x64
    model.add(layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_regularizer=reg))
    model.add(layers.BatchNormalization(momentum=bn_momentum))
    model.add(layers.ReLU())
    if deep:
        model.add(layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_regularizer=reg))
        model.add(layers.BatchNormalization(momentum=bn_momentum))
        model.add(layers.ReLU())
    model.add(layers.MaxPooling2D())

    # Block 3: -> 28x28x128
    model.add(layers.Conv2D(128, 3, padding="same", use_bias=False, kernel_regularizer=reg))
    model.add(layers.BatchNormalization(momentum=bn_momentum))
    model.add(layers.ReLU())
    if deep:
        model.add(layers.Conv2D(128, 3, padding="same", use_bias=False, kernel_regularizer=reg))
        model.add(layers.BatchNormalization(momentum=bn_momentum))
        model.add(layers.ReLU())
    model.add(layers.MaxPooling2D())

    # Block 4: -> 14x14x256
    model.add(layers.Conv2D(256, 3, padding="same", use_bias=False, kernel_regularizer=reg))
    model.add(layers.BatchNormalization(momentum=bn_momentum))
    model.add(layers.ReLU())
    if deep:
        model.add(layers.Conv2D(256, 3, padding="same", use_bias=False, kernel_regularizer=reg))
        model.add(layers.BatchNormalization(momentum=bn_momentum))
        model.add(layers.ReLU())
    model.add(layers.MaxPooling2D())

    # Classifier: average of each filter -> 256 numbers -> one probability per landmark
    model.add(layers.GlobalAveragePooling2D())
    if dropout:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_classes, activation="softmax", kernel_regularizer=reg))
    return model
