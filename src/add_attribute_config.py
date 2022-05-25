import argparse
from config import TriTrainingConfig
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()

    for p in (Path(args.dir) / 'logs').glob('*/*'):
        if (p / 'tritraining_validation_scores.txt').exists():
            print(p)
            cfg = TriTrainingConfig(p / 'config.yml')
            #cfg.tritraining.data_prep_done = True
            cfg.tritraining.tritraining_done = True
            cfg.save()