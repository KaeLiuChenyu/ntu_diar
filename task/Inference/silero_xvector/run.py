import os
import sys
import torch

from hyperpyyaml import load_hyperpyyaml

from infer_scr.infer_silero_xvector import InferTask


if __name__ == "__main__":

    # Load config.yaml
    print("Load hparams")
    with open(sys.argv[1]) as fin:
        hparams = load_hyperpyyaml(fin)

    print("Initial inference")
    infer_task = InferTask(hparams)

    print("Start infering")
    infer_task.infer(
      wav_file = hparams["input_audio"],
      outfile = hparams["pre_rttm"],
    )

    
    