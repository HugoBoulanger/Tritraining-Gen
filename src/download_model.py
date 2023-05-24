from pathlib import Path
from utils import download_model, download_gpt2, download_t5

# Download and save locally both the model and the tokenizer from huggingface.co
#download_model(save_dir=Path('../examples/bert-large/'), name='bert-large')
#download_model(save_dir=Path('../examples/biobert-base-cased/'), name='biobert-base-cased-v1.1')
#download_t5(save_dir=Path('../examples/google/t5-v1_1-large'), name="google/t5-v1_1-large")
download_gpt2(save_dir=Path('../examples/gpt2-large/'), name='gpt2-large')