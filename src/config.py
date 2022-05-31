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

from pathlib import Path, PosixPath
from typing import Union, Text

import yaml

from box import Box
from utils import fix_seed


class Config:
    """
    Dot-based access to configuration parameters saved in a YAML file.
    """
    def __init__(self, file: Union[Path, Text]):
        """
        Load the parameters from the YAML file.
        If no path are given in the YAML file for bert_checkpoint and seqeval, the corresponding objects will be load
        if used (needs an internet connection).
        """
        self.file = file
        # get a Box object from the YAML file
        with open(str(file), 'r') as ymlfile:
            cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

        # manually populate the current Config object with the Box object (since Box inheritance fails)
        for key in cfg.keys():
            setattr(self, key, getattr(cfg, key))


        # resolve seqeval config into a name or a path
        seqeval_path = getattr(self.model, "seqeval_path", None)
        self.model.seqeval_path = file.parent / Path(seqeval_path) if seqeval_path is not None else 'seqeval'

        patience = getattr(self.model, "patience", None)
        self.model.patience = patience if patience is not None else 20

        self.dataset.path = file.parent / Path(self.dataset.path)

        # Don't lowercase if the corresponding attribute is not defined in config.yml
        self.dataset.do_lowercase = getattr(self.dataset, 'do_lowercase', False)

        # Correct types in train (ex. lr = 5e-5 is read as string)
        for float_var in ["dropout", "learning_rate", "slot_loss_coeff"]:
            val = getattr(self.train, float_var)
            if type(val) != float:
                setattr(self.train, float_var, float(val))

        assert self.train.validation_metric in ["intent_acc", "slot_f1", "loss"], "Unrecognized validation metric"

        # Some attributes could not be defined in config.yml, set them as None
        self.train.num_workers = getattr(self.train, "num_workers", None)
        self.train.seed = getattr(self.train, "seed", None)

        # Fix seed if specified
        if self.train.seed is not None:
            fix_seed(self.train.seed)

        if (file.parent / Path(self.model.name_or_path)).exists():
            self.model.name_or_path = file.parent / Path(self.model.name_or_path)

    def save(self):
        # print([a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))])
        d = {}
        for attr, value in self.__dict__.items():
            print(attr, value, type(value))
            if type(value) == Box:
                d[attr] = {}
                for k, v in value.items():
                    if type(v) in [Path, PosixPath]:
                        d[attr][k] = str(v.relative_to(self.file.parent))
                    else:
                        d[attr][k] = v

        cfg_w = Box(d)
        cfg_w.to_yaml(filename=self.file)



class TriTrainingConfig:
    """
        Dot-based access to configuration parameters saved in a YAML file.
    """

    def __init__(self, file: Union[Path, Text]):
        """
        Load the parameters from the YAML file.
        If no path are given in the YAML file for bert_checkpoint and seqeval, the corresponding objects will be load
        if used (needs an internet connection).
        """
        self.file = file
        # get a Box object from the YAML file
        with open(str(file), 'r') as ymlfile:
            cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)


        # manually populate the current Config object with the Box object (since Box inheritance fails)
        for key in cfg.keys():
            setattr(self, key, getattr(cfg, key))


        self.preprocess.path = file.parent / Path(self.preprocess.path)
        self.dataset.path = file.parent / Path(self.dataset.path)
        self.preprocess.gen_path = file.parent / Path(self.preprocess.gen_path)
        if (file.parent / Path(self.model.name_or_path)).exists():
            self.model.name_or_path = file.parent / Path(self.model.name_or_path)

        self.train.num_workers = getattr(self.train, "num_workers", None)

        append_unlabeled = getattr(self.tritraining, "append_unlabeled", None)
        self.tritraining.append_unlabeled = append_unlabeled if append_unlabeled is not None else False

        seqeval_path = getattr(self.model, "seqeval_path", None)
        self.model.seqeval_path = file.parent / Path(seqeval_path) if seqeval_path is not None else 'seqeval'

    def save(self):
        #print([a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))])
        d = {}
        for attr, value in self.__dict__.items():
            print(attr, value, type(value))
            if type(value) == Box:
                d[attr] = {}
                for k, v in value.items():
                    if type(v) in [Path, PosixPath]:
                        d[attr][k] = str(v.relative_to(self.file.parent))
                    else:
                        d[attr][k] = v

        cfg_w = Box(d)
        cfg_w.to_yaml(filename=self.file)

