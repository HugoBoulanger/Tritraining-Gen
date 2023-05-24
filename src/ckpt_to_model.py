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
    try:
        model = MultiBERTForNLU.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            map_location='cpu',
            dataset=dataset,
            config=config
        )
    except Exception as e:
        print(e)
        exit(-1)
    torch.save(model, str(ckpt_path)[:-4] + 'pt')
    with open(args.dir / 'logs/checkpoints/best_model_path.txt', 'w') as f:
        f.write(str(ckpt_path)[:-4] + 'pt')
    os.remove(str(ckpt_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="Experiment directory where config.yml is located")
    args = parser.parse_args()

    args.dir = Path(args.dir)
    ckpt_to_model(args)