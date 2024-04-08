from pathlib import Path

import torch
import torch.nn as nn

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from .config_parser import load_gpt_2_config, GPT_2_Config


def get_all_text(data_path: Path):
    data = Path(data_path)
    for file in data.glob("*.txt"):
        with file.open("r") as f:
            text = f.read()
        yield (text)


def get_or_build_tokenizer(config: GPT_2_Config, data_path: Path) -> Tokenizer:
    tokenizer_path = config.checkpoints_folder / config.tokenizer_file
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_text(data_path), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
