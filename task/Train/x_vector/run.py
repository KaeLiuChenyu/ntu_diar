import os
import sys
import torch

from hyperpyyaml import load_hyperpyyaml
from ntu_diar.scr.utils.parse_arg import parse_arguments

from train_scr.train_x_vectors import TrainTask
from utils.dataio_prep import dataio_prep


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    # Load config.yaml
    hparams_file, run_opts, overrides = parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    print("Load hparams")
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    print("Load datasets")
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    if not os.path.isdir(hparams["output_folder"]):
        os.makedirs(hparams["output_folder"])

    # Task initialization
    print("Initial training")
    train_task = TrainTask(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    print("Start training")
    train_task.fit(
        train_task.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )