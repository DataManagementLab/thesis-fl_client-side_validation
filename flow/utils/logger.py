from os import times
from pandas.core.frame import DataFrame
import yaml, csv, shutil, json
from yaml import Loader, Dumper
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Iterator
import pandas as pd
from copy import deepcopy

from matplotlib.pyplot import savefig
from matplotlib.figure import Figure

import torch
from torch.nn import Module

from .time_tracker import TimeTracker

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
    BASE_PATH = 'logs'
    CONFIG_FILE = 'config.yml'
    STATS_FILE = 'stats.csv'
    EPOCH_DIR = 'epoch'
    METRICS_DIR = 'metrics'
    PLOTS_DIR = 'plots'
    TIMES_FILE = 'times.json'
    MODEL_FILE = 'model.pth'
    ATTACKS_APPLIED_LOG = 'attacks_applied.csv'
    ATTACKS_DETECTED_LOG = 'attacks_detected.csv'
    EXP_NAME_FORMAT = 'experiment_%Y_%m_%d__%H:%M'

    def __init__(self, base_path: str = None, exp_name: str = None, timestamp: datetime = None) -> None:
        assert not (exp_name and timestamp), 'Logger does not accept timestamp and exp_name parameter simultaneously for initialization.'
        self.base_path = Path(base_path if base_path else self.BASE_PATH)
        if exp_name:
            self.exp_str = exp_name
        else:
            timestamp = timestamp or datetime.now()
            self.exp_str = timestamp.strftime(self.EXP_NAME_FORMAT)
        self.get_path().mkdir(parents=True, exist_ok=True)
    
    def clone(self):
        return deepcopy(self)

    def get_path(self, epoch: int = None, tail: str = '') -> Path:
        path = self.base_path / self.exp_str
        if epoch is not None:
            path = path / self.EPOCH_DIR / f'{epoch:03}'
        return path / tail
    
    def find_paths(self, pattern: str = '*', epoch: int = None, tail: str = '', recursive=False):
        if recursive:
            return self.get_path(epoch, tail).rglob(pattern)
        else:
            return self.get_path(epoch, tail).glob(pattern)

    @property
    def num_epochs(self) -> int:
        return len(list(self.get_path(tail=self.EPOCH_DIR).glob('*')))
    
    def epoch_exists(self, epoch):
        return self.get_path(epoch).is_dir()

    def save_model(self, epoch: int, model: Module) -> None:
        self.get_path(epoch).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.get_path(epoch, self.MODEL_FILE))

    def load_model(self, epoch: int, model: Module) -> Module:
        if not self.get_path(epoch, self.MODEL_FILE).exists(): raise FileExistsError(f'No file {self.MODEL_FILE} can be loaded for epoch {epoch} in {self.exp_str}')
        model.load_state_dict(torch.load(self.get_path(epoch, self.MODEL_FILE)))
        return model
    
    def set_lock(self, lock):
        self.lock = lock
    
    def rem_lock(self):
        del self.lock
    
    def has_lock(self):
        return hasattr(self, 'lock')
    
    def acquire_lock(self):
        if self.has_lock(): self.lock.acquire()
    
    def release_lock(self):
        if self.has_lock(): self.lock.release()
    
    def init_attack_application_log(self, epoch):
        data = ['epoch', 'batch', 'element', 'type']
        path = self.get_path(epoch, self.ATTACKS_APPLIED_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.acquire_lock()
        try:
            self._log_csv(path, data)
        finally:
            self.release_lock()

    def log_attack_application(self, epoch, batch, element, attack_type):
        path = self.get_path(epoch, self.ATTACKS_APPLIED_LOG)
        if not path.is_file(): self.init_attack_application_log(epoch)
        self.acquire_lock()
        try:
            self._log_csv(path, [epoch, batch, element, attack_type])
        finally:
            self.release_lock()
    
    def init_attack_detection_log(self, epoch):
        data = ['epoch', 'batch', 'element', 'type']
        path = self.get_path(epoch, self.ATTACKS_DETECTED_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.acquire_lock()
        try:
            self._log_csv(path, data)
        finally:
            self.release_lock()

    def log_attack_detection(self, epoch, batch, element, attack_type):
        path = self.get_path(epoch, self.ATTACKS_DETECTED_LOG)
        if not path.is_file(): self.init_attack_detection_log(epoch)
        self.acquire_lock()
        try:
            self._log_csv(path, [epoch, batch, element, attack_type])
        finally:
            self.release_lock()
    
    def check_attack_detection(self):
        join_index = ['epoch', 'batch', 'element']
        for epoch in range(1, self.num_epochs):
            applications = self._load_csv(self.ATTACKS_APPLIED_LOG, epoch).set_index(join_index)
            detections = self._load_csv(self.ATTACKS_DETECTED_LOG, epoch).set_index(join_index)
            TP = applications.join(
                detections,
                lsuffix='_app',
                rsuffix='_det')
            FN = applications.drop(TP.index)
            FP = detections.drop(TP.index)
            print('epoch', f'{epoch:03}', end='\t')
            print(f'Detected: {len(TP)}/{len(applications)}', end='\t')
            print('TP', len(TP), end='\t')
            print('FP', len(FP), end='\t')
            print('FN', len(FN))
    
    def save_config(self, config: Dict) -> None:
        self._save_yaml(config, self.CONFIG_FILE)
    
    def save_times(self, times: TimeTracker, epoch: int) -> None:
        self._save_json(times.get_dict(), self.TIMES_FILE, epoch)
    
    def load_times(self, epoch: int) -> None:
        return TimeTracker.from_dict(self._load_json(self.TIMES_FILE, epoch))
    
    def copy_config(self, config_file: Path, force=False) -> None:
        assert config_file.is_file(), f'Config file that is supposed to be copied does not exist. {config_file}'
        config_dest = self.get_path(tail=self.CONFIG_FILE)
        if not force:
            assert not config_dest.is_file(), f'Config file destination already esists. Set parameter force=True to force overwrite. {config_dest}'
        shutil.copy(config_file, config_dest)
    
    def load_config(self, *element_list) -> Iterator:
        config = self._load_yaml(self.CONFIG_FILE)
        if len(element_list) == 0: return config
        else: return (config[el] for el in element_list)

    def save_stats(self, data: pd.DataFrame):
        self._save_csv(data, self.STATS_FILE, index_label='epoch')
    
    def load_stats(self) -> pd.DataFrame:
        return self._load_csv(self.STATS_FILE)

    def save_metrics(self, label: str, metrics: pd.DataFrame):
        metrics_file = Path(self.METRICS_DIR) / f'{label}.csv'
        assert not self.get_path(tail=metrics_file).exists(), f'Metrics file for label "{label}" already exists.'
        self.get_path(tail=metrics_file).parent.mkdir(parents=True, exist_ok=True)
        self._save_csv(metrics, metrics_file, index_label='epoch')

    def load_metrics(self, label: str) -> pd.DataFrame:
        metrics_file = Path(self.METRICS_DIR) / f'{label}.csv'
        return self._load_csv(metrics_file)

    def list_metrics(self):
        metrics_files = self.find_paths(tail=self.METRICS_DIR)
        return sorted([ label.stem for label in metrics_files ])

    def clear_metrics(self):
        metrics_files = self.find_paths(tail=self.METRICS_DIR)
        for metric_file in metrics_files: metric_file.unlink()

    def save_plot(self, metric: str, figure: Figure, extension: str = 'png'):
        plot_path = self.get_path(tail=self.PLOTS_DIR) / f'{metric}.{extension}'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_path)

    def clear_plots(self):
        plots_files = self.find_paths(tail=self.PLOTS_DIR)
        for plot_file in plots_files: plot_file.unlink()
    
    # Default methods for saving and loading
    
    def _save_json(self, obj, path, epoch=None) -> None:
        path = self.get_path(epoch, tail=path)
        with open(path, 'w') as f:
            json.dump(obj, f)

    def _load_json(self, path, epoch=None) -> Dict:
        path = self.get_path(epoch, tail=path)
        assert path.is_file(), f'File {path} does not exist.'
        with open(path, 'r') as f:
            return json.load(f)
    
    def _save_yaml(self, obj, path, epoch=None) -> None:
        path = self.get_path(epoch, tail=path)
        with open(path, 'w') as f:
            yaml.dump(obj, f, Dumper)

    def _load_yaml(self, path, epoch=None) -> Dict:
        path = self.get_path(epoch, tail=path)
        assert path.is_file(), f'File {path} does not exist.'
        with open(path, 'r') as f:
            return yaml.load(f, Loader)
    
    def _save_csv(self, data_frame: DataFrame, path, epoch=None, **kwargs) -> None:
        path = self.get_path(epoch, tail=path)
        data_frame.to_csv(path, **kwargs)

    def _load_csv(self, path, epoch=None, **kwargs) -> DataFrame:
        path = self.get_path(epoch, tail=path)
        assert path.is_file(), f'File {path} does not exist.'
        return pd.read_csv(path, **kwargs)
    
    def _log_json(self, path, data: Dict) -> None:
        with open(path, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    def _log_csv(self, path, data: List) -> None:
        with open(path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)
