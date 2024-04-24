from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import Dataset


def get_all_text(data_path: Path):
    data = Path(data_path)
    for file in data.glob("*.txt"):
        with file.open("r") as f:
            text = f.read()
        yield (text)


def get_or_build_tokenizer(data_path: Path, tokenizer_path: Path) -> Tokenizer:
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


class GPT2Dataset(Dataset):
    def __init__(self, seq_len: int, data_path: Path, tokenizer: Tokenizer) -> None:
        all_text = "\n".join(get_all_text(data_path))
        self._seq_len = seq_len
        self._encoded_text = tokenizer.encode(all_text).ids
        self._pad_token = torch.tensor(
            [tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self) -> int:
        return len(self._encoded_text) - self._seq_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        # input = torch.Tensor(self._encoded_text[idx : idx + self._seq_len])
        input = torch.from_numpy(
            np.array(self._encoded_text[idx : idx + self._seq_len])
        )
        mask = (input != self._pad_token).unsqueeze(0).int() & causal_mask(
            input.size(0)
        )
        return {"input": input, "mask": mask}


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
