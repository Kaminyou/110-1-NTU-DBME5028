"""Run fitting with pixels vector
python ./scripts/run_pixels_fit.py \
    --model xgb
"""
import sys
import argparse
import re
import numpy as np
from collections import defaultdict
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(".")
from src.utils import load_image_targets, get_result_metrics
from src.models import Trainer


def make_inputs(inputs):
    inputs = inputs.reshape((len(inputs), -1))
    inputs = np.float32(inputs) / 255.
    return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="logistic",
        help="Model Type, support [logistic|xgb]"
    )
    parser.add_argument(
        "--data_root",
        default="./data/",
        help="Where is the data"
    )
    args = parser.parse_args()
    print(args)

    files = glob("{}/*[x|y].h5".format(args.data_root))
    filepath = defaultdict(dict)
    for i in files:
        file_ = Path(i).stem
        condition = re.findall(pattern="train|valid|test", string=file_)[0]
        target_type = re.findall(pattern="_[x|y]", string=file_)[0]
        filepath[condition][target_type] = i

    ## ONLY FOR COURSE DEMO (NORMALLY, YOU SHOULD USE TRAIN FILES)
    train_x, train_y = load_image_targets(
        filepath["valid"]["_x"],
        filepath["valid"]["_y"]
    )
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x,
        train_y,
        test_size=0.1
    )

    test_x, test_y = load_image_targets(
        filepath["test"]["_x"],
        filepath["test"]["_y"]
    )

    train_x_vector = make_inputs(train_x)
    valid_x_vector = make_inputs(valid_x)
    test_x_vector = make_inputs(test_x)

    trainer = Trainer(method=args.model)
    trainer.set_train_data(train_x_vector, train_y)
    trainer.set_valid_data(valid_x_vector, valid_y)

    print("Train Start")
    trainer.train()
    y_pred = trainer.predict(test_x_vector)

    results = get_result_metrics(
        y_true=test_y,
        y_score=y_pred
    )
    print(results)
