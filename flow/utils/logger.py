import yaml, csv, shutil
from yaml import Loader, Dumper
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterator

import torch
from torch.nn import Module

class Logger:
    """
    Logger for Experiment.

    experiment_2021_05_01__12:23
        epoch_0 (initial)
        epoch_1
            model.pth
        stats.csv (created with plotter)
            accuracy_total
            accuracy_per_number
            precision & recall
    """
    BASE_PATH = "logs"
    CONFIG_FILE = "config.yml"
    STATS_FILE = "stats.csv"

    def __init__(self, base_path: str = None, timestamp: datetime = datetime.now()) -> None:
        self.base_path = Path(base_path if base_path else self.BASE_PATH)
        timestamp_str = timestamp.strftime("%Y_%m_%d__%H:%M") # %Y_%m_%d__%H:%M
        self.exp_str = f"experiment_{timestamp_str}"
        self.get_path().mkdir(parents=True, exist_ok=True)

    def get_path(self, epoch: int = None, tail: str = '') -> Path:
        path = self.base_path / self.exp_str
        if epoch is not None:
            path = path / f'epoch_{epoch}'
        return path / tail
    
    def find_paths(self, pattern: str = '*', epoch: int = None, recursive=False):
        if recursive:
            return self.get_path(epoch).rglob(pattern)
        else:
            return self.get_path(epoch).glob(pattern)


    @property
    def num_epochs(self) -> int:
        return len(list(self.get_path().glob('epoch_*')))

    def save_model(self, epoch: int, model: Module) -> None:
        self.get_path(epoch).mkdir(parents=True, exist_ok=False)
        torch.save(model.state_dict(), self.get_path(epoch, 'model.pth'))

    def load_model(self, epoch: int, model: Module) -> Module:
        if not self.get_path(epoch, 'model.pth').exists(): raise FileExistsError(f'No model.pth can be loaded for epoch {epoch} in {self.exp_str}')
        model.load_state_dict(torch.load(self.get_path(epoch, 'model.pth')))
        return model
    
    def save_config(self, config: Dict) -> None:
        with open(self.get_path(tail=self.CONFIG_FILE), 'w') as f:
            yaml.dump(config, f, Dumper)
    
    def copy_config(self, config_file: Path) -> None:
        assert config_file.exists() and config_file.is_file()
        shutil.copy(config_file, self.get_path(tail=self.CONFIG_FILE))
    
    def load_config(self) -> Dict:
        with open(self.get_path(tail=self.CONFIG_FILE), 'r') as f:
            return yaml.load(f, Loader)
    
    def put_csv(self, file_path: Path, columns: Iterator[Iterator[str]], delimiter: str = ',') -> None:
        with open(self.get_path() / file_path, 'a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=delimiter)
            csv_writer.writerows(columns)



        

