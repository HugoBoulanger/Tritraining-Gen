from pathlib import Path
from utils import download_model, download_gpt2

# Download and save locally both the model and the tokenizer from huggingface.co
download_model(save_dir=Path('../examples/bert-large-cased/'), name='bert-large-cased')
#download_gpt2(save_dir=Path('../examples/gpt2/'), name='gpt2')