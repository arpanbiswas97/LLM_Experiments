from pathlib import Path

from pydantic import BaseModel


class GPT2Config(BaseModel):
    model_name: str = "gpt2"
    tokenizer_path: Path = Path("CHECKPOINTS/gpt2_tokenizer")
    train_path: Path = Path("TRAIN_DATA")
    test_path: Path = Path("TEST_DATA")
    model_weights_path: Path = Path("CHECKPOINTS/gpt2")
    epochs: int = 100
    batch_size: int = 40
    seq_len: int = 40
    d_model: int = 512
    d_mlp: int = 2048
    heads: int = 4
    N: int = 4
    dropout: float = 0.9
    preload: str = "latest"
    lr: float = 0.1
