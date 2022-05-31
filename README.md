# Generating unlabelled data for a tri-training approach in a low resourced NER task
Companion repository for the paper "Generating unlabelled data for a tri-training approach in a low resourced NER task".

This project contains all the scripts and information needed to reproduce the experiments presented in the paper.

## Citation

```bibtex
Paper accepted at Deeplo 2022 (bibtex not final).
@inproceedings{boulanger2022trigen,
    title={Generating unlabelled data for a tri-training approach in a low resourced NER task},
    author={Boulanger, Hugo and Lavergne, Thomas and Rosset, Sophie},
    year={2022}
}


```

## Installation

1) Create `conda` environment:

```shell
conda create -n trigen python==3.8
conda activate trigen
```

2) Install PyTorch (>= 1.7.1) following the instructions of the [docs](https://pytorch.org/get-started/locally/#start-locally)

```shell
conda install pytorch -c pytorch
conda install torchmetrics pytorch-lightning -c conda-forge
```

3) Install dependencies:
```shell
pip install -r requirements.txt
```

## Requirements

### Configuration file

Each experiment needs a YAML configuration file including various paths and hyper-parameters.

`example_config.yml` is an example of a configuration file.

### Dataset

To run the experiment you need to download the CoNLL and I2B2 datasets.

After downloading the dataset, the path to the directory containing the train, dev, test splits must be (e.g.
`train_EN.tsv`, the naming scheme is a remnant of the original repo and purpose) indicated in the configuration file:
```yaml
dataset:
  # Path to TSV files for train, dev and test
  path: '/path/to/tsv/files'
```

### Offline mode (no internet connection)

If you try to run an experiment in **offline mode**, it will fail when trying to download the 
configured BERT, the associated tokenizer and the seqeval script.
To avoid this, do the following:
1. Pre-download the chosen model ('bert-base-multilingual-cased' is the one used in the paper) and the associated tokenizer (needs an internet connexion):
```python
from pathlib import Path
from utils import download_model

# Download and save locally both the model and the tokenizer from huggingface.co
download_model(save_dir=Path('/path/to/model/directory'), name='bert-base-multilingual-cased')
```
2. Get [seqeval.py](https://github.com/huggingface/datasets/blob/master/metrics/seqeval/seqeval.py) and save it locally. 
3. Modify `config.yml` with the paths to the dowloaded BERT model and `seqeval.py`:
```yaml
model:
  name_or_path: '/path/to/model/directory'

seqeval_path: '/path/to/seqeval.py'
```


## Run the experiments

The experiments with the configurations used for Deeplo 2022 can be found in ~/examples.
First build the experiments using the next command from the ~/src directory

```
python make_tri_training.py ../examples/name_of_the_experiment ./default_config.yml 
```
To run the CoNLL and I2B2 experiments on SLURM use the ~/src/run_conll.sh and ~/run_i2b2.sh scripts.



## License

Licence of current modifications

```
MIT License

Copyright (c) 2022 Université Paris-Saclay
Copyright (c) 2022 Laboratoire Interdisciplinaire des Sciences du Numérique (LISN)
Copyright (c) 2022 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


Licence of initial repo

```
MIT License

Copyright (c) 2021 Université Paris-Saclay
Copyright (c) 2021 Laboratoire national de métrologie et d'essais (LNE)
Copyright (c) 2021 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```