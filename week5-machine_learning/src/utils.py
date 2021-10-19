"""utils.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    confusion_matrix,
    recall_score,
    precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

import h5py


def load_h5df(filepath):
    with h5py.File(filepath, "r") as f:
        key = list(f.keys())[0]
        data = list(f.get(key))
    return data


def load_image_targets(input_path, target_path):
    inputs = load_h5df(input_path)  # list of array
    targets = load_h5df(target_path)  # list of array

    return np.array(inputs), np.concatenate(targets).ravel().astype(np.float32)


def show_example_images(image_array, figsize=(8, 8), n_grid_x=10):
    n_images = len(image_array)
    n_grid_y = (n_images // n_grid_x)

    fig = plt.figure(figsize=figsize)
    for counter, img in enumerate(image_array):
        if (counter >= n_grid_x * n_grid_y):
            continue
        fig.add_subplot(n_grid_y, n_grid_x, counter + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_result_metrics(y_true, y_score, score_threshold=0.5):
    # For binary cases
    y_pred_binary = y_score >= score_threshold

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
    auc_score = auc(fpr, tpr)

    f1sc = f1_score(y_true=y_true, y_pred=y_pred_binary)
    cm_ = confusion_matrix(y_true=y_true, y_pred=y_pred_binary)
    recall_ = recall_score(y_true=y_true, y_pred=y_pred_binary)
    precision_ = precision_score(y_true=y_true, y_pred=y_pred_binary)

    output = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_score,
        "f1_score": f1sc,
        "confusion_matrix": cm_,
        "recall": recall_,
        "precision": precision_
    }
    return output


def load_and_process_digits():
    digits_data = load_digits()
    x, y = digits_data["images"], digits_data["target"]
    x = x.reshape((len(x), -1))  # convert to vector
    x = np.array(x, dtype=np.float32) / 255.  # do min/max normalization
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    return x_train, y_train, x_valid, y_valid, x_test, y_test
