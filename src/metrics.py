# GAP (Global Average Precision), the metric of the Kaggle competition
import numpy as np


# 1. every image gets one prediction (most probable landmark) and its confidence
# 2. sort them from most to least confident
# 3. GAP = (1/M) * sum of P(i) * rel(i)
#    P(i): precision of the first i predictions, rel(i): 1 if prediction i is right
def gap(y_true, probs):
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    order = np.argsort(-conf, kind="stable")
    rel = (pred == y_true)[order].astype(float)
    precision = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    return (precision * rel).sum() / len(y_true)


# check:  python -m src.metrics
if __name__ == "__main__":
    y = np.array([0, 1, 2])
    right = np.eye(3) * 0.9 + 0.05
    assert gap(y, right) == 1.0                  # all right -> 1
    assert gap(y, right[[1, 2, 0]]) == 0.0       # all wrong -> 0
    # 2 of 3 right: being unsure on the wrong one scores higher
    good = np.array([[.9, .1, 0], [0, .8, .2], [.6, .4, 0]])
    bad = np.array([[.6, .4, 0], [0, .7, .3], [.9, .1, 0]])
    assert np.isclose(gap(y, good), (1 + 1) / 3)
    assert np.isclose(gap(y, bad), (1 / 2 + 2 / 3) / 3)
    print("gap ok")
