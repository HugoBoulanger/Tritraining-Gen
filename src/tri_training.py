#MIT License
#
#Copyright (c) 2022 Université Paris-Saclay
#Copyright (c) 2022 Laboratoire Interdisciplinaire des Sciences du Numérique (LISN)
#Copyright (c) 2022 CNRS
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



# MIT License
#
# Copyright (c) 2021 Université Paris-Saclay
# Copyright (c) 2021 Laboratoire national de métrologie et d'essais (LNE)
# Copyright (c) 2021 CNRS
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
import pandas as pd
import random
import shutil, sys, os
import re
import datetime

import pytorch_lightning as pl
import torch as t
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import pipeline, set_seed, GPT2TokenizerFast, GPT2LMHeadModel

from shutil import rmtree
from config import TriTrainingConfig, Config
from pathlib import Path
from model import MultiBERTForNLU, MultiBERTTokenizer
from dataset import MultiATIS, TriTrainingData, LabelEncoding
from utils import get_checkpoint_path
from metrics import SlotF1
from tqdm import tqdm


def train_BERT(args):
    """
    capsulated version of train_iid.py
    :return:
    """
    log_dir = args.dir
    #print(args)
    if args.from_ckpt:
        #ckpt_path = get_checkpoint_path(args.dir)
        with open(log_dir / 'logs/checkpoints/best_model_path.txt', 'r') as f:
            ckpt_path = Path(f.read())
        """
        recovery_path = ckpt_path.parent / f"recovery_{args.epochs}_epochs_{ckpt_path.name[:-5]}"
        if recovery_path.exists():
            print(f"Removing existing directory {recovery_path.name}")
            rmtree(recovery_path)
        recovery_path.mkdir(exist_ok=False)
        log_dir = recovery_path
        """

    config = Config(args.dir / "config.yml")
    args.epochs = config.train.epochs_per_lang
    languages = config.train.languages
    monolingual = len(languages) == 1 or args.from_ckpt

    print("Loading dataset... ", end="", flush=True)
    dataset = MultiATIS(config, MultiBERTTokenizer)
    print("OK")
    config.train.languages = languages[0]
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
        dirpath=None
    )

    early_stopping = EarlyStopping(
        monitor=f"dev_{validation_metric}",
        mode="min" if validation_metric == "loss" else "max",
        patience=config.model.patience,
        min_delta=0.00001
    )


    print(f"train log_dir : {log_dir}")
    # The checkpoints and logging files are automatically saved in save_dir
    logger = TensorBoardLogger(save_dir=log_dir, name=None, version='logs')
    epochs = config.train.epochs_per_lang if not args.from_ckpt else args.epochs
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=epochs,
        num_sanity_val_steps=0,
        logger=logger,
        checkpoint_callback=True,
        move_metrics_to_cpu=True,
        callbacks=[checkpoint_callback, early_stopping]
    )

    #print(*[m for m in model.named_modules()], sep='\n')

    trainer.fit(model, train_loader, val_loader)

    # EVALUATION
    eval_split = 'test'
    print(f"Evaluate the trained model on {eval_split}")

    # Load the best checkpoint
    print("Loading best checkpoint...   ", end="", flush=True)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"train best_ckpt : {best_ckpt}")
    with open(log_dir / 'logs/checkpoints/best_model_path.txt', 'w') as f:
        f.write(best_ckpt)
    model = MultiBERTForNLU.load_from_checkpoint(
        checkpoint_path=best_ckpt,
        map_location=model.device,
        dataset=dataset,
        config=model.cfg
    )
    print("OK")

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
        results = trainer.test(model, eval_loader, verbose=False)
        report = model.test_slot_f1.compute()['report']
        print(results)
        print(report)
        #results = report
        #report = results["slot_filling_report"]
        performance[lang.upper()] = results[0]["test_f1"]
        report.to_csv(Path(logger.log_dir) / f"slot_filling_report_{eval_split}_{lang.upper()}.csv")

    print()
    print(pd.DataFrame(performance, index=["slot f1"]).round(2))
    print()

    # Delete checkpoint
    if not config.train.keep_checkpoints:
        Path(best_ckpt).unlink()

    for p in (log_dir / "logs/checkpoints/").glob('*.ckpt'):
        if str(p) != str(best_ckpt):
            p.unlink()

    del trainer
    del model

def load_model(args):
    """
    loads a modle from a checkpoint
    :param args:
    :return:
    """
    print(f"load_model args.dir : {args.dir}")
    #ckpt_path = get_checkpoint_path(args.dir)
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
        map_location='cuda' if args.gpus > 0 else 'cpu',
        dataset=dataset,
        config=config
    )
    return model


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

def annotate_w_BERT(args, model, dataset, name, config):
    """
    Capsulated version of annotate.py
    :return:
    """

    loader = dataset.get_loader(
        split="unlabeled",
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=0,
        num_sanity_val_steps=0,
    )

    print("####   Annotation    ####\n\n")
    with t.no_grad():
        pred = trainer.predict(model, loader)

        preds = []
        k = 0
        for x in loader:
            new_preds = pred[k].get_predictions(log=False)[0]
            values, indices = t.max(new_preds, dim=2)
            align = t.clip(x['slot_labels'], max=0)
            indices = indices + align
            values = list(values.numpy())
            indices = list(indices.numpy())
            nex = [[(values[i][j], indices[i][j]) for j in range(len(values[i]))] for i in range(len(values))]
            preds.extend(nex)
            k+=1

        del pred

        for i in range(len(preds)):
            preds[i] = remove_missaligned(preds[i])

        labels = [dataset.label_encoding.decodify_slot_labels([preds[i][j][1] for j in range(len(preds[i]))]) for i in
                  range(len(preds))]
        db = pd.DataFrame(
            [[dataset.unlabeled['utterance'][i], ' '.join(labels[i]), '0']
             for i in range(len(preds))], columns=['utterance', 'slot_labels', 'intent'])

        print(f"annotate args.dir : {args.dir}")
        db.to_csv(args.dir / f"../data/{name}.tsv", sep='\t')
        #s = pd_to_column(db)
        #with open(args.dir / f"data/{name}.iob", 'w') as f:
        #    f.write(s)



def evaluate_BERT(args, model, loader):
    """
    Evaluates BERT
    :return: 
    """

    log_dir = args.dir
    print(f"evaluate log_dir : {log_dir}")
    """
    recovery_path = ckpt_path.parent / f"recovery_{args.epochs}_epochs_{ckpt_path.name[:-5]}"
    if recovery_path.exists():
        print(f"Removing existing directory {recovery_path.name}")
        rmtree(recovery_path)
    recovery_path.mkdir(exist_ok=False)
    log_dir = recovery_path
    """

    logger = TensorBoardLogger(save_dir=log_dir, name=None, version='logs')
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=1,
        num_sanity_val_steps=0,
        logger=logger,
        checkpoint_callback=False,
        callbacks=[]
    )

    results = trainer.test(model, loader, verbose=False)[0]
    report = model.test_slot_f1.compute()['report']
    print(results)
    print(report)
    print(f"evaluate log_dir : {log_dir}")
    return results["test_f1"]


def bootstrap_sampling(lines, rand):
    """
    :return:
    """
    tri1 = rand.choices(lines, k=len(lines))
    tri2 = rand.choices(lines, k=len(lines))
    tri3 = rand.choices(lines, k=len(lines))
    return tri1, tri2, tri3


def tri_training_pretrain(args):
    """
    Initializes the three taggers pretrain the models on the split data
    :return:
    """
    print(args.dir)
    cfg = TriTrainingConfig(args.dir / 'config.yml')
    dataset = TriTrainingData(cfg, LabelEncoding(args.dir / 'data'), MultiBERTTokenizer(cfg))

    #print(dataset.unlabeled)

    for i in range(1, 4):
        print(f"########## model {i} #########\n\n\n")
        dir = args.dir
        args.dir = args.dir / f"tri-{i}"
        args.from_ckpt = False
        train_BERT(args)
        model = load_model(args)
        cfg = Config(args.dir / 'config.yml')
        annotate_w_BERT(args, model, dataset, f"tri-{i}", cfg)
        args.dir = dir
        shutil.copyfile(args.dir / f"data/tri-{i}/train_EN.tsv", args.dir / f"data/tri-{i}/train_bootstrap_EN.tsv")
        print(f"########## End of model {i} #########\n\n\n")



def tri_training_episode(args, cfg):
    """
    Runs one episode of Tri-training.
    Takes current data splits in entry.
    writes new data

    Main block of tri_training
    :return:
    """
    scores = []

    print(f"Episode args.dir : {args.dir}")
    dataset = TriTrainingData(cfg, LabelEncoding(args.dir / 'data'), MultiBERTTokenizer(cfg))

    for i in range(1, 4):
        dir = args.dir
        args.dir = args.dir / f"tri-{i}"
        print(f"args.dir : {args.dir}")
        if i < cfg.tritraining.current_tri:
            cfg_loc = Config(args.dir / 'config.yml')
            model = load_model(args)
            loader = dataset.get_loader(
                split="val",
                batch_size=cfg_loc.train.batch_size,
                shuffle=False,
                num_workers=cfg_loc.train.num_workers
            )
            scores.append(evaluate_BERT(args, model, loader))
            args.dir = dir
            continue
        print(f"########## model {i} #########\n\n\n")
        train_BERT(args)
        cfg_loc = Config(args.dir / 'config.yml')
        model = load_model(args)
        annotate_w_BERT(args, model, dataset, f"tri-{i}", cfg_loc)
        loader = dataset.get_loader(
            split="val",
            batch_size=cfg_loc.train.batch_size,
            shuffle=False,
            num_workers=cfg_loc.train.num_workers
        )
        scores.append(evaluate_BERT(args, model, loader))
        args.dir = dir
        cfg.tritraining.current_tri += 1
        cfg.save()
        print(f"########## End of model {i} #########\n\n\n")
    cfg.tritraining.current_tri = 1
    cfg.save()
    return scores

def is_empty(line):
    l = line.split(sep='\t')[2]
    ls = l.split()
    for t in ls:
        if t not in ['O', 'o']:
            return False
    return True

def make_pseudolabels_set(args):
    """
    Creates each train set for the next step
    :param args:
    :return:
    """
    ids = [1, 2, 3]
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        shutil.copyfile(args.dir / f"data/train_EN.tsv", args.dir / f"data/tri-{ids[i]}/train_EN.tsv")
        with open(args.dir / f"data/tri-{ids[j]}.tsv", 'r') as fj:
            lj = fj.readlines()[1:]
        with open(args.dir / f"data/tri-{ids[k]}.tsv", 'r') as fk:
            lk = fk.readlines()[1:]
        with open(args.dir / f"data/tri-{ids[i]}/train_EN.tsv", 'r') as f:
            new_lines = f.readlines()
        c = 0
        for l in range(len(lj)):
            if lj[l] == lk[l] and not is_empty(lj[l]):
                new_lines.append(lj[l])
                c += 1
        with open(args.dir / f"data/tri-{ids[i]}/train_EN.tsv", 'w') as f:
            f.writelines(new_lines)
        print(f"Pseudo-labels / possible sentences : {c} / {len(lj)}")


def save_checkpoints():
    """
    Moves checkpoints and data associated with one of the tri-training taggers to a save location
    :return:
    """
    ...

def text_cleaning(line):
    """
    takes a line of text and cleans it
    """
    line = re.subn('\\n', ' ', line)[0]
    line = re.subn(r' +', ' ', line)[0]
    return line

def generate_follow_ups(generator, sentence):
    """
    generates follow-up sentences using GPT2
    :return:
    """

    s = sentence.split()[1:-1]
    #print(s)
    sj = ' '.join(s)
    g = generator(sj, max_len=len(s)*3, num_return_sequences=5, return_full_text=False)
    #print(g)
    #print(*g, sep='\n')
    ret = [text_cleaning(gg['generated_text'].encode().decode('utf-8', 'ignore')) for gg in g]
    #print(ret)
    return ret

def generate_end(generator, sentence, percentage):
    """
    Generates end of sentences using GPT2
    cuts the sentences to appropriate percentage
    :return:
    """
    s = sentence.split()[1:-1]
    s = s[:max(int(len(s)*percentage), 1)]
    sj = ' '.join(s)
    g = generator(sj, max_len=int(len(s) * 2 / percentage), num_return_sequences=5, return_full_text=True)
    ret = [text_cleaning(gg['generated_text'].encode().decode('utf-8', 'ignore')) for gg in g]
    # print(ret)
    return ret

def text_to_pd(lines):
    """
    creates a panda with empty labels, adds BOS, EOS
    """
    new_lines = []

    for l in lines:
        sl = "BOS " + ' '.join(l.split()) + " EOS"
        new_lines.append([sl, ' '.join(['O' for i in range(len(sl.split()))]), '0'])

    db = pd.DataFrame(new_lines, columns=['utterance', 'slot_labels', 'intent'])
    return db


def remove_empty(l):
    new_l = []
    for s in l:
        if len(s.split()) > 0:
            new_l.append(s)
    return new_l

def generate(train_path, seed, path_out, gen_path, append_to_unlabeled=False):
    """
    Generate the full data using GPT2
    :param args:
    :return:
    """
    set_seed(seed)
    generator = pipeline('text-generation', model=GPT2LMHeadModel.from_pretrained(gen_path), tokenizer=GPT2TokenizerFast.from_pretrained(gen_path), binary_output=True)
    data = list(pd.read_csv(train_path, sep='\t', usecols=['utterance', 'slot_labels', 'intent'])['utterance'])
    new_data = []
    if append_to_unlabeled:
        ul = list(pd.read_csv(path_out, sep='\t', usecols=['utterance', 'slot_labels', 'intent'])['utterance'])
        data.extend(ul)
        new_data.extend(ul)

    for d in data:
        new_data.extend(generate_follow_ups(generator, d))
        new_data.extend(generate_end(generator, d, 0.75))
        new_data.extend(generate_end(generator, d, 0.5))
        new_data.extend(generate_end(generator, d, 0.25))

    new_data = remove_empty(new_data)

    df = text_to_pd(new_data)
    df.to_csv(path_out, sep='\t', encoding='utf-8')


def tri_training(args):
    """
    Full tri-training procedure
    :return:
    """

    cfg = TriTrainingConfig(args.dir / 'config.yml')
    dir = args.dir

    if not cfg.tritraining.pretrain_done:
        tri_training_pretrain(args)
        args.dir = dir
        make_pseudolabels_set(args)
        args.dir = dir
        cfg.tritraining.pretrain_done = True
        cfg.save()

    episodes = cfg.tritraining.episodes
    scores = [[-1], [-1], [-1]]
    args.from_ckpt = True
    writer1 = SummaryWriter(f"{dir}/runs/{datetime.datetime.now():%y%m%d%H%M}_Tritraining_1_{cfg.preprocess.seed}")
    writer2 = SummaryWriter(f"{dir}/runs/{datetime.datetime.now():%y%m%d%H%M}_Tritraining_2_{cfg.preprocess.seed}")
    writer3 = SummaryWriter(f"{dir}/runs/{datetime.datetime.now():%y%m%d%H%M}_Tritraining_3_{cfg.preprocess.seed}")
    b = cfg.tritraining.current_episode
    for e in range(b, episodes):
        print(f'\n\n\n###### Beginning Episode {e} / {episodes} ######\n\n\n')
        args.dir = dir
        cfg.tritraining.current_episode = e
        cfg.save()
        #print(f"args.dir : {args.dir}")
        s = tri_training_episode(args, cfg)
        cfg.tritraining.current_episode = e+1
        cfg.save()
        writer1.add_scalar(f"Tritraining/Validation_F1", s[0], e)
        writer2.add_scalar(f"Tritraining/Validation_F1", s[1], e)
        writer3.add_scalar(f"Tritraining/Validation_F1", s[2], e)
        if s[0] <= max(scores[0]) and s[1] <= max(scores[1]) and s[2] <= max(scores[2]):
            scores[0].append(s[0])
            scores[1].append(s[1])
            scores[2].append(s[2])
            with open(args.dir / Path('tritraining_validation_scores.txt'), 'w') as f:
                f.write(str(scores))
            break
        scores[0].append(s[0])
        scores[1].append(s[1])
        scores[2].append(s[2])
        make_pseudolabels_set(args)
        print(f'\n\n\n###### End Episode {e} / {episodes} ######\n\n\n')

    cfg = TriTrainingConfig(args.dir / 'config.yml')

    dataset = TriTrainingData(cfg, LabelEncoding(args.dir / 'data'), MultiBERTTokenizer(cfg))
    loader = dataset.get_loader(
        split="test",
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers
    )

    scores = []
    for i in range(1, 4):
        args.dir = dir / f"tri-{i}"
        model = load_model(args)
        scores.append(evaluate_BERT(args, model, loader))
    with open(args.dir / "scores.txt", 'w') as f:
        f.write(str(scores))

    models = []
    for i in range(1, 4):
        args.dir = dir / f"tri-{i}"
        models.append(load_model(args))
    args.dir = dir
    evaluate_tri_training(args, models, dataset, 'results_tri_training', loader, cfg)




def threshold():
    """
    Threshold under which predictions are not considered during tri-training
    :return:
    """
    ...

def two_votes():
    """
    takes 2 series of predictions with probabilities and returns a series of prediction

    :return:
    """
    ...

def evaluate_tri_training(args, models, dataset, name, loader, cfg):
    """
    Capsulated version of annotate.py
    :return:
    """
    #print(cfg.model.seqeval_path)
    #print(dataset.config.model.seqeval_path)
    metr = SlotF1(dataset.label_encoding, ignore_index=dataset.config.dataset.ignore_index, name_or_path=str(dataset.config.model.seqeval_path), compute_report=True)

    preds = []

    with t.no_grad():

        for x in tqdm(loader):
            # print(data_eval.label_encoding.decodify_slot_labels(x['slot_labels'][0].numpy()))
            new_preds = []
            for m in models:
                new_preds.append(m(x).get_predictions(log=False)[0])
            new_preds = t.sum(t.stack(new_preds, dim=0), dim=0)
            metr.update(new_preds, x['slot_labels'])

            values, indices = t.max(new_preds, dim=2)
            align = t.clip(x['slot_labels'], max=0)
            indices = indices + align
            values = list(values.numpy())
            indices = list(indices.numpy())
            nex = [[(values[i][j], indices[i][j]) for j in range(len(values[i]))] for i in range(len(values))]


        """    
        for i in range(len(preds)):
            preds[i] = remove_missaligned(preds[i])

        pred_labels = [dataset.label_encoding.decodify_slot_labels([preds[i][j][1] for j in range(len(preds[i]))]) for i in
                  range(len(preds))]

        results = Seqeval(pred_labels, dataset.test['slot_labels'])
        """
        dico = metr.compute()
        print(dico['f1'])
        dico['report'].to_csv(args.dir / f"results_{name}.csv")




def copy_split_half_tsv(path_in, path_out1, path_out2):
    with open(path_in, 'r') as f:
        lines = f.readlines()
    head = [lines[0]]
    with open(path_out1, 'w') as f:
        f.writelines(head + lines[1:len(lines)//2])
    with open(path_out2, 'w') as f:
        f.writelines(head + lines[len(lines)//2:])

def prepare_data(args):
    """
    Takes the train, dev, and test.
    Copies dev and test to experiment directory.
    Splits the train (seed of the whole)
    :return:
    """
    cfg = TriTrainingConfig(args.dir / "config.yml")

    try:
        os.mkdir(args.dir / "data")
    except Exception:
        ...
    finally:
        shutil.copyfile(cfg.preprocess.path / "test_EN.tsv", args.dir / "data/test_EN.tsv")
        shutil.copyfile(cfg.preprocess.path / "dev_EN.tsv", args.dir / "data/dev_EN.tsv")

    try:
        os.mkdir(args.dir / "data/tri-1")
        os.mkdir(args.dir / "data/tri-2")
        os.mkdir(args.dir / "data/tri-3")
    except Exception:
        ...
    finally:
        shutil.copyfile(cfg.preprocess.path / "test_EN.tsv", args.dir / "data/tri-1/test_EN.tsv")
        copy_split_half_tsv(cfg.preprocess.path / "dev_EN.tsv", args.dir / "data/tri-1/dev_EN.tsv", args.dir / "data/tri-1/val_EN.tsv")
        shutil.copyfile(cfg.preprocess.path / "test_EN.tsv", args.dir / "data/tri-2/test_EN.tsv")
        copy_split_half_tsv(cfg.preprocess.path / "dev_EN.tsv", args.dir / "data/tri-2/dev_EN.tsv", args.dir / "data/tri-2/val_EN.tsv")
        shutil.copyfile(cfg.preprocess.path / "test_EN.tsv", args.dir / "data/tri-3/test_EN.tsv")
        copy_split_half_tsv(cfg.preprocess.path / "dev_EN.tsv", args.dir / "data/tri-3/dev_EN.tsv", args.dir / "data/tri-3/val_EN.tsv")
        shutil.copyfile(args.dir / "data/tri-3/val_EN.tsv", args.dir / "data/val_EN.tsv")
    rand = random.Random()
    rand.seed(a=cfg.preprocess.seed)

    with open(cfg.preprocess.path / "train_EN.tsv", 'r') as f:
        lines = f.readlines()
    new_lines = [lines[0]]
    lines = lines[1:]
    rand.shuffle(lines)
    unlabeled = lines[cfg.preprocess.size:]
    if cfg.tritraining.append_unlabeled:
        with open(args.dir / "data/unlabeled.tsv", 'w') as f:
            f.writelines([new_lines[0]] + unlabeled)
    lines = lines[:cfg.preprocess.size]                       # A bit useless but will stop splits higher than the size
    new_lines.extend(lines[:cfg.preprocess.split])

    tris = bootstrap_sampling(lines[:cfg.preprocess.split], rand)

    with open(args.dir / "data/train_EN.tsv", 'w') as f:
        f.writelines(new_lines)

    for i, t in enumerate(tris):
        #print(*t)
        with open(args.dir / f"data/tri-{i+1}/train_EN.tsv", 'w') as f:
            f.writelines([new_lines[0]] + t)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="Experiment directory where config.yml is located")
    parser.add_argument('--gpus', type=int, required=False, default=1, help="Number of gpus to train on")
    parser.add_argument('--epochs', type=int, required=False, default=1)
    args = parser.parse_args()
    args.dir = Path(args.dir)

    #DATA PREPARATION

    cfg = TriTrainingConfig(args.dir / 'config.yml')

    if not cfg.tritraining.data_prep_done:
        prepare_data(args)
        cfg.tritraining.data_prep_done = True
        cfg.save()


    #TRAIN BASELINE

    dir = args.dir
    if not cfg.tritraining.baseline_done:
        args.dir = args.dir / "baseline"
        args.from_ckpt = False
        train_BERT(args)
        cfg.tritraining.baseline_done = True
        cfg.save()

    args.dir = dir

    if not cfg.tritraining.generation_done:
        generate(args.dir / "data/train_EN.tsv", cfg.preprocess.seed, args.dir / "data/unlabeled.tsv", cfg.preprocess.gen_path, append_to_unlabeled=cfg.tritraining.append_unlabeled)
        cfg.tritraining.generation_done = True
        cfg.save()

    #TRI_TRAINING

    args.dir = dir
    cfg = TriTrainingConfig(args.dir / 'config.yml')
    if not cfg.tritraining.tritraining_done:
        tri_training(args)
        cfg = TriTrainingConfig(args.dir / 'config.yml')
        cfg.tritraining.tritraining_done = True
        cfg.save()
    else:
        dataset = TriTrainingData(cfg, LabelEncoding(args.dir / 'data'), MultiBERTTokenizer(cfg))
        loader = dataset.get_loader(
            split="test",
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers
        )
        models = []
        for i in range(1, 4):
            args.dir = dir / f"tri-{i}"
            models.append(load_model(args))
        args.dir = dir
        evaluate_tri_training(args, models, dataset, 'results_tri_training', loader, cfg)
