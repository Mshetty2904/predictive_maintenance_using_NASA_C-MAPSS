import numpy as np


def nasa_score(y_true, y_pred):
    diff = np.asarray(y_pred) - np.asarray(y_true)
    score = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1,
        np.exp(diff / 10.0) - 1,
    )
    return np.sum(score)
