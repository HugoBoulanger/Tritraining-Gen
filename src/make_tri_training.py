import yaml
import argparse
import shutil
import copy

from pathlib import Path, PosixPath
from box import Box



def merge_dicts(base, override):
    for k, v in override.items():
        if type(v) == dict:
            if type(base[k]) == dict:
                merge_dicts(base[k], v)
            else:
                base[k] = v
        else:
            base[k] = v



def make_training(path, default_config, config_chg):
    """
    Makes a directory with a config file for training a BERT tagger
    :return:
    """
    print(f"Making {path} directory")
    path.mkdir(parents=True, exist_ok=False)
    print("Ok")



    print(f"Making {path}/config.yml")
    def_dict = copy.deepcopy(default_config)
    cfg_chg = copy.deepcopy(config_chg)

    #print(def_dict)

    for k, v in def_dict.items():
        for kk, vv in v.items():
            if type(vv) in [Path, PosixPath]:
                def_dict[k][kk] = str(vv)

    for k, v in cfg_chg.items():
        for kk, vv in v.items():
            if type(vv) in [Path, PosixPath]:
                cfg_chg[k][kk] = str(vv)

    merge_dicts(def_dict, cfg_chg)
    cfg_w = Box(def_dict)
    cfg_w.to_yaml(filename=path / 'config.yml')
    print(f"Ok")


def make_one_directory(path, default_configs, edits):
    """
    Creates the directory, containing the config file for one tritraining
    :return:
    """
    ed_dict = copy.deepcopy(edits)
    make_training(path / 'baseline', default_configs['BERT'], ed_dict['BERT'])

    for i in range(1, 4):
        ed_dict['BERT']['dataset'] = {}
        ed_dict['BERT']['dataset']['path'] = Path(f'../data/tri-{i}')
        make_training(path / f'tri-{i}', default_configs['BERT'], ed_dict['BERT'])

    print(f"Making {path}/config.yml")
    def_dict = copy.deepcopy(default_configs)
    ed_dict['tri-train']['dataset'] = {}
    ed_dict['tri-train']['dataset']['path'] = "./data/"
    merge_dicts(def_dict['tri-train'], ed_dict['tri-train'])
    cfg_w = Box(def_dict)
    cfg_w['tri-train'].to_yaml(filename=path / 'config.yml')
    print(f"Ok")

def loop_on_variables(path_to_exproot, path_from_exproot, default_configs, edits):
    """
    Recursive loop on edits dictionnary, each list of parameters become a new directory layer and a new loop
    :return:
    """
    path_to_exproot = path_to_exproot / '..'
    bottom = True
    for k, v in edits['tri-train'].items():
        for kk, vv in v.items():
            if type(vv) == list:
                bottom = False
                new_edits = copy.deepcopy(edits)
                for e in vv:
                    p = path_from_exproot / f"{kk}={e}"
                    p.mkdir(parents=True, exist_ok=False)
                    new_edits['tri-train'][k][kk] = e
                    loop_on_variables(path_to_exproot, p, default_configs, new_edits)
                break

    if bottom:
        nedits = copy.deepcopy(edits)
        ndefault_configs = copy.deepcopy(default_configs)
        for k, v in nedits['tri-train'].items():
            for kk, vv in v.items():
                if type(vv) == str or type(vv) == Path:
                    if (path_from_exproot / path_to_exproot / vv).exists():
                        nedits['tri-train'][k][kk] = str(path_to_exproot / vv)
        for k, v in ndefault_configs['tri-train'].items():
            for kk, vv in v.items():
                if type(vv) == str or type(vv) == Path:
                    if (path_from_exproot / path_to_exproot / vv).exists():
                        ndefault_configs['tri-train'][k][kk] = str(path_to_exproot / vv)

        path_to_exproot_b = path_to_exproot / '..'
        for k, v in nedits['BERT'].items():
            for kk, vv in v.items():
                if type(vv) == str or type(vv) == Path:
                    if (path_from_exproot / path_to_exproot / vv).exists():
                        nedits['BERT'][k][kk] = path_to_exproot_b / vv
        for k, v in ndefault_configs['BERT'].items():
            for kk, vv in v.items():
                if type(vv) == str or type(vv) == Path:
                    if (path_from_exproot / path_to_exproot / vv).exists():
                        ndefault_configs['BERT'][k][kk] = path_to_exproot_b / vv

        #print(type(ndefault_configs))
        ndefault_configs['BERT']['dataset']['path'] = Path('../data')
        make_one_directory(path_from_exproot, ndefault_configs, nedits)


def make_experiment(path, default_cfg):
    """
    Takes a directory containing a config file config.yml and seqeval.py (Huggingface)
    and creates the experiment structure from the config file
    :return:
    """
    print(f"Making {path}/ directory")
    logs = path / 'logs'
    logs.mkdir(parents=True, exist_ok=False)
    print("Ok")

    with open(path / "config.yml", 'r') as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None).to_dict()
    print(cfg)
    loop_on_variables(Path('.'), logs, default_cfg, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='path to directory containing config.yml and seqeval.py')
    parser.add_argument('default_path', type=str, help='path to default config for this experiment')
    args = parser.parse_args()

    with open(Path(args.default_path), 'r') as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None).to_dict()
    print(cfg)
    make_experiment(Path(args.dir), cfg)
