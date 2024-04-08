from pathlib import Path

import yaml
from pydantic import BaseModel

CONFIG_PATH = "GPT_2/config.yaml"


class GPT_2_Config(BaseModel):
    checkpoints_folder: Path
    tokenizer_file: str


def load_gpt_2_config():
    config_file_path = Path(CONFIG_PATH)
    with config_file_path.open("r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)
    config = GPT_2_Config(**config_dict)
    return config
