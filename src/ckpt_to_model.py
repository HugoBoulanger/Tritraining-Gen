# MIT License
#
# Copyright (c) 2022 Université Paris-Saclay
# Copyright (c) 2022 Laboratoire Interdisciplinaire des Sciences du Numérique (LISN)
# Copyright (c) 2022 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import pytorch_lightning as pl
import torch
import shutil, sys, os
import torch.nn as nn

from model import MultiBERTForNLU, MultiBERTTokenizer
from dataset import MultiATIS, TriTrainingData, LabelEncoding
from config import TriTrainingConfig, Config
from pathlib import Path


def ckpt_to_model(args):
    with open(args.dir / 'logs/checkpoints/best_model_path.txt', 'r') as f:
        ckpt_path = Path(f.read())
    print(f"ckpt_path: {ckpt_path}")
    config = Config(args.dir / "config.yml")

    languages = config.train.languages
    monolingual = len(languages) == 1 or args.from_ckpt

    print("Loading dataset... ", end="", flush=True)
    dataset = MultiATIS(config, MultiBERTTokenizer)
    print("OK")

    print(f"Loading model from checkpoint {str(ckpt_path)}... ", end="", flush=True)
    model = MultiBERTForNLU.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        map_location='cpu',
        dataset=dataset,
        config=config
    )
    torch.save(model, (ckpt_path - 'ckpt') + 'pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="Experiment directory where config.yml is located")
    args = parser.parse_args()

    args.dir = Path(args.dir)
    ckpt_to_model(args)