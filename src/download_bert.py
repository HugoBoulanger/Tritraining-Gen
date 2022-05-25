from pathlib import Path
from utils import download_model, download_gpt2

# Download and save locally both the model and the tokenizer from huggingface.co
download_model(save_dir=Path('../bert-base-uncased/'), name='bert-base-uncased')
#download_gpt2(save_dir=Path('../distilgpt2/'), name='distilgpt2')