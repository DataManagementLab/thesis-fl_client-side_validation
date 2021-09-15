from pathlib import Path
from numpy import imag
from torch.nn.modules.module import T
from torch.utils.data import DataLoader
from typing import List, Optional
import sys, yaml
from yaml import Loader
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from flow.utils import Logger
from flow import models, datasets, utils

class Plotter:

    PLOT_DIR = 'plots'
    METRICS = ['TP', 'TN', 'FP', 'FN', 'precision', 'recall', 'f1_score', 'accuracy', 'average_loss', 'overall_accuracy']

    def __init__(self, logger: Logger):
        self.logger = logger

    def generate(self):

        config = self.logger.load_config()
        model = getattr(models, config['model']['type'])(**config['model']['params'])
        loss_fn = getattr(torch.nn, config['loss_fn']['type'])()
        dataset: DataLoader = getattr(datasets, 'get_dataloader_{}'.format(config['dataset']['type']))(**config['dataset']['params_test'])
        test_labels = torch.unique(dataset.dataset.targets).tolist()
        metrics_csv = { x: pd.DataFrame(columns=self.METRICS) for x in test_labels }

        for epoch in range(self.logger.num_epochs):
            model = self.logger.load_model(epoch, model)
            model.eval()

            label_confusion_matrix = { x: dict(TP=0, TN=0, FP=0, FN=0) for x in test_labels }
            print(f'epoch: {epoch}', end='\t')

            with torch.no_grad():
                average_loss = 0.0
                overall_accuracy = 0.0
                for batch, (data, target) in enumerate(dataset):
                    output = model(data)
                    loss = loss_fn(output, target)
                    average_loss += loss.item()
                    pred = torch.argmax(output, dim=1)
                    for p, t in zip(pred.tolist(), target.tolist()):
                        if p == t: # TP / TN
                            for x in label_confusion_matrix.keys():
                                if x == p:
                                    label_confusion_matrix[x]['TP'] += 1
                                    overall_accuracy += 1
                                # else:
                                #     label_confusion_matrix[x]['TN'] += 1
                        elif p != t: # FP / FN / TN
                            for x in label_confusion_matrix.keys():
                                if x == p:
                                    label_confusion_matrix[x]['FP'] += 1
                                if x == t:
                                    label_confusion_matrix[x]['FN'] += 1
                                else:
                                    label_confusion_matrix[x]['TN'] += 1
                
                average_loss /= len(dataset)
                overall_accuracy /= len(dataset) * dataset.batch_size

                print(f'average_loss: {average_loss}')

                for x, v in label_confusion_matrix.items():
                    TP = v['TP']; TN = v['TN']; FP = v['FP']; FN = v['FN']
                    
                    precision = self._calc_precision(TP, FP)
                    recall = self._calc_recall(TP, FN)

                    csv_tuple = dict(
                        TP=TP,
                        TN=TN,
                        FP=FP,
                        FN=FN,
                        precision=precision,
                        recall=recall,
                        f1_score=self._calc_f1_score(precision, recall),
                        accuracy=self._calc_accuracy(TP, TN, FP, FN),
                        average_loss=average_loss, 
                        overall_accuracy=overall_accuracy)
                    metrics_csv[x] = metrics_csv[x].append(csv_tuple, ignore_index=True)

        for x, v in metrics_csv.items():
            metrics_file = self.metrics_path / f'{x}_metrics.csv'
            assert not metrics_file.exists()
            v.to_csv(metrics_file, index_label='epoch')
    
    def plot_metric(self, metric: str, ymin: int = None, ymax: int = None, xmin: int = None, xmax: int = None, x_label: str = 'epochs'):
        print(f'Generating plot for metric "{metric}"', end=' ')
        fig, ax = plt.subplots()
        metric_files = self.logger.find_paths('*_metrics.csv')
        for metric_file in sorted(metric_files):
            key = metric_file.name.split('_')[0]
            df = pd.read_csv(metric_file)
            df[metric].plot(ax=ax, label=key)
        ax.legend()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.grid()
        self.plots_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plots_path / f'{metric}.png')
        print(utils.vc(True))
    
    @classmethod
    def plot_summary(cls, plot_cnf: Path) -> None:
        assert plot_cnf.exists() and plot_cnf.is_file(), 'Summary plot config file does not exist.'
        with open(plot_cnf, 'r') as f:
            plot = yaml.load(f, Loader)
        log_dir = Path(plot['log_dir'])
        plot_path = log_dir / plot['plot_dir'] / '{}.png'.format(plot['name'])
        fig, ax = plt.subplots()
        for el in plot['data']:
            metric_file = log_dir / el['experiment'] / '{}_metrics.csv'.format(el['class'])
            df = pd.read_csv(metric_file)
            df[el['metric']].plot(ax=ax, label=el['label'])
        ax.legend()
        if 'xlabel' in plot: ax.set_xlabel(plot['xlabel'])
        if 'ylabel' in plot: ax.set_ylabel(plot['ylabel'])
        if 'grid' in plot and plot['grid'] == True: ax.grid()
        xmin = plot['xmin'] if 'xmin' in plot else None
        xmax = plot['xmax'] if 'xmax' in plot else None
        ymin = plot['ymin'] if 'ymin' in plot else None
        ymax = plot['ymax'] if 'ymax' in plot else None
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
    
    def clean_metrics(self):
        metric_files = self.logger.find_paths('*_metrics.csv')
        for metric_file in metric_files: metric_file.unlink()

    def clean_plots(self):
        metric_files = self.logger.find_paths('*.png', recursive=True)
        for metric_file in metric_files: metric_file.unlink()

    def _calc_precision(self, TP: int, FP: int) -> Optional[float]:
        if not TP == 0: return TP / (TP + FP)

    def _calc_recall(self, TP: int, FN: int) -> Optional[float]:
        if not TP == 0: return TP / (TP + FN)
    
    def _calc_f1_score(self, precision: Optional[float], recall: Optional[float]) -> Optional[float]:
        if precision and recall and not precision == recall == 0: 
            return 2. * precision * recall / (precision + recall)

    def _calc_accuracy(self, TP: int, TN: int, FP: int, FN: int) -> Optional[float]:
        if not TP == TN == 0: return (TP + TN) / (TP + TN + FP + FN)
    
    @property
    def plots_path(self) -> Path:
        return self.logger.get_path(tail=self.PLOT_DIR)
    
    @property
    def metrics_path(self) -> Path:
        return self.logger.get_path()