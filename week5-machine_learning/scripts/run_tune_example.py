"""
python ./scripts/run_tune_example.py

# To see results
from ray.tune import Analysis
analysis = Analysis(PATH_TO_EXP_DIR)
df = analysis.trial_dataframes

"""
import sys
import os
import numpy as np
from random import shuffle
from collections import deque
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch import optim
from ray import tune

sys.path.append(".")
from src.utils import load_and_process_digits
from src.models import LogisticRegressionTorch


def simple_loader(inputs, targets, batch_size=128, shuffle_per_iteration=20):
    index = 0
    while True:
        indexes_get = np.arange(index * batch_size, (index + 1) * batch_size) % len(inputs)
        x_ = np.take(inputs, indexes_get, axis=0)
        y_ = np.take(targets, indexes_get, axis=0)

        index += 1
        if index % shuffle_per_iteration == 0:
            full_index = np.arange(len(x_))
            shuffle(full_index)
            inputs = np.take(inputs, full_index, axis=0)
            targets = np.take(targets, full_index, axis=0)
        yield x_, y_

def train_digits(config: dict):
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_and_process_digits()
    train_loader = simple_loader(x_train, y_train, batch_size=config["batch_size"])

    model = LogisticRegressionTorch(input_dim=x_train.shape[-1], output_dim=10)

    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    train_losses, valid_losses = [], []
    bst_loss = 1e+4
    patient_counter = 0
    for i_epoch in range(config["num_epochs"]):
        loss_record = deque(maxlen=100)
        for _ in range(len(x_train) // config["batch_size"]):
            x, y = next(train_loader)

            logits = model(torch.from_numpy(x))
            loss_train = loss_fn(logits, torch.from_numpy(y))

            ### Do regularization
            if config["l1_alpha"] > 0:
                l1_term = torch.tensor(0.)
                for model_params in model.parameters():
                    reg = torch.abs(model_params).sum()
                    l1_term += reg
                loss_train = loss_train + config["l1_alpha"] * l1_term

            if config["l2_alpha"] > 0:
                l2_term = torch.tensor(0.)
                for model_params in model.parameters():
                    reg = torch.norm(model_params)
                    l2_term += reg
                loss_train = loss_train + config["l2_alpha"] * l2_term

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            loss_record.append(loss_train.detach().cpu().numpy())

        with torch.no_grad():
            yp_logits = model(torch.from_numpy(x_valid))
            loss_valid = loss_fn(yp_logits, torch.from_numpy(y_valid))
            loss_valid = loss_valid.detach().cpu().numpy()

        print("Epoch: {}/{}, Training Loss: {:.3f}, Validation Loss: {:.3f}".format(
            str(i_epoch + 1).zfill(4),
            config["num_epochs"],
            np.mean(loss_record),
            loss_valid
        ), flush=True, end="\r")
        train_losses.append(np.mean(loss_record))
        valid_losses.append(loss_valid)

        tune.report(validation_loss=loss_valid)  # validation_loss can be keywords you want

        ### Do earlystopping
        if patient_counter >= config["n_earlystopping_rounds"]:
            return model, train_losses, valid_losses

        if loss_valid < bst_loss:
            bst_loss = loss_valid
            patient_counter = 0
        else:
            patient_counter += 1

    return model, train_losses, valid_losses


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int = 500
    l1_alpha: float = 0.
    l2_alpha: float = 0.
    n_earlystopping_rounds: int = 1e+8

    def to_dict(self):
        return asdict(self)


if __name__ == "__main__":
    # Force use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    train_config = TrainConfig(
        batch_size=tune.choice([64, 128]),
        learning_rate=tune.grid_search([0.5, 1, 1.5]),
        num_epochs=1000,
        l1_alpha=tune.grid_search([0, 0.001, 0.01]),
        l2_alpha=tune.grid_search([0, 0.001, 0.01]),
        # n_earlystopping_rounds
    )

    analysis = tune.run(
        train_digits,
        config=train_config.to_dict(),
        num_samples=3,
        progress_reporter=tune.CLIReporter(max_error_rows=20)
    )  # Total num_trials = num_samples**tunable_params
