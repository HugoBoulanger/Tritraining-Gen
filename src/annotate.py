import torch as t
import argparse
import pandas as pd
from pathlib import Path
from shutil import rmtree

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import MultiATIS
from model import MultiBERTForNLU, MultiBERTTokenizer
from config import Config
from utils import get_checkpoint_path


def remove_missaligned(line):
    return [t for t in line if t[1] >= 0]

def pd_to_column(db):
    s = ""

    for i in range(len(db['ref'])):
        ut = db['utterance'][i].split()
        ref = db['ref'][i].split()
        hyp = db['hyp'][i].split()
        for j in range(1, len(ut) - 1):
            s += f"{ut[j]} {ref[j]} {hyp[j]}\n"
        s +="\n"
    return s

def eval():
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="Experiment directory where config.yml is located")
    parser.add_argument('--gpus', type=int, required=False, default=1, help="Number of gpus to train on")
    parser.add_argument('--from_ckpt', action='store_true', help="Whether to train from a checkpoint")
    parser.add_argument('--epochs', type=int, required=False, default=1, help="Number of training epochs")
    args = parser.parse_args()
    args.dir = Path(args.dir)

    log_dir = args.dir
    if args.from_ckpt:
        ckpt_path = get_checkpoint_path(args.dir)
        recovery_path = ckpt_path.parent / f"recovery_{args.epochs}_epochs_{ckpt_path.name[:-5]}"
        if recovery_path.exists():
            print(f"Removing existing directory {recovery_path.name}")
            rmtree(recovery_path)
        recovery_path.mkdir(exist_ok=False)
        log_dir = recovery_path

    config = Config(args.dir / "config.yml")

    languages = config.train.languages
    monolingual = len(languages) == 1 or args.from_ckpt

    print("Loading dataset... ", end="", flush=True)
    dataset = MultiATIS(config, MultiBERTTokenizer)
    print("OK")

    if args.from_ckpt:
        print(f"Loading model from checkpoint {str(ckpt_path)}... ", end="", flush=True)
        model = MultiBERTForNLU.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            dataset=dataset,
            config=config
        )
    else:
        print("Loading model... ", end="", flush=True)
        model = MultiBERTForNLU(dataset, config)
    print("OK")

    # TRAINING
    train_lang = languages[0] if monolingual else 'all'
    print("Will train on:", train_lang)
    data_training = dataset[train_lang]
    train_loader = data_training.get_loader(
        split='train',
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers
    )
    val_loader = data_training.get_loader(
        split='dev',
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers
    )

    # Only save the best model according to the validation metric
    validation_metric = config.train.validation_metric
    checkpoint_callback = ModelCheckpoint(
        monitor=f"dev_{validation_metric}",
        mode="min" if validation_metric == "loss" else "max",
        save_top_k=1,
        dirpath=log_dir if args.from_ckpt else None
    )

    # The checkpoints and logging files are automatically saved in save_dir
    logger = TensorBoardLogger(save_dir=log_dir, name=None, version='logs')
    epochs = config.train.epochs_per_lang if not args.from_ckpt else args.epochs
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=epochs,
        num_sanity_val_steps=0,
        logger=logger,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback]
    )


    # EVALUATION
    eval_split = 'test'
    print(f"Evaluate the trained model on {eval_split}")

    if not monolingual:
        languages.append('all')

    performance = {}
    for lang in languages:
        # Load the split to test the trained model
        data_eval = dataset[lang] if lang != 'all' else data_training
        eval_loader = data_eval.get_loader(
            split=eval_split,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers
        )

        # Get slot-filling scores report with scores per slot types
        print(f"Type eval loader : {type(eval_loader)}")
        #results = trainer.test(model, eval_loader, verbose=False)[0]
        #print(results)
        #report = results["slot_filling_report"]
        #performance[lang.upper()] = results["slot_filling_f1"]
        #report.to_csv(Path(logger.log_dir) / f"slot_filling_report_{eval_split}_{lang.upper()}.csv")

        print(data_eval.label_encoding.slot_to_int)

        preds = []

        print(eval_loader)

        with t.no_grad():

            for x in eval_loader:

                #print(data_eval.label_encoding.decodify_slot_labels(x['slot_labels'][0].numpy()))
                new_preds = model(x).get_predictions(log=False)[0]
                values, indices = t.max(new_preds, dim=2)
                align = t.clip(x['slot_labels'], max=0)
                indices = indices + align
                values = list(values.numpy())
                indices = list(indices.numpy())
                nex = [[(values[i][j], indices[i][j]) for j in range(len(values[i]))] for i in range(len(values))]
                preds.extend(nex)


            for i in range(len(preds)):
                preds[i] = remove_missaligned(preds[i])

            labels = [data_eval.label_encoding.decodify_slot_labels([preds[i][j][1] for j in range(len(preds[i]))])  for i in range(len(preds))]
            db = pd.DataFrame([[data_training.test['utterance'][i], ' '.join(data_training.test['slot_labels'][i]), ' '.join(labels[i])] for i in range(len(preds))], columns=['utterance', 'ref', 'hyp'])
            db.to_csv(f"../20220304_conll/test.tsv", sep='\t')
            s = pd_to_column(db)
            with open(f"../20220304_conll/test.iob", 'w') as f:
                f.write(s)

    print()
    #print(pd.DataFrame(performance, index=["slot f1"]).round(2))
    print()

