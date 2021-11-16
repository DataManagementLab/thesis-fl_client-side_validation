from collections import defaultdict
from pathlib import Path
from numpy import imag, zeros
from torch.nn.modules.module import T
from torch.utils.data import DataLoader
from typing import List, Optional
import sys, yaml, time
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

        for label, metrics in metrics_csv.items():
            self.logger.save_metrics(label, metrics)
    
    def plot_metric(self, metric: str, ymin: int = None, ymax: int = None, xmin: int = None, xmax: int = None, x_label: str = 'epochs'):
        print(f'Generating plot for metric "{metric}"', end=' ')
        fig, ax = plt.subplots()
        for label in self.logger.list_metrics():
            df = self.logger.load_metrics(label)
            df[metric].plot(ax=ax, label=label)
        ax.legend()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.grid()
        self.logger.save_plot(metric, fig)
        plt.close(fig)
        print(utils.vc(True))
    
    @classmethod
    def plot_summary(cls, plot_cnf: Path) -> None:
        assert plot_cnf.is_file(), 'Summary plot config file does not exist.'
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
        if plot.get('grid') == True: ax.grid()
        # if 'xlabel' in plot: ax.set_xlabel(plot['xlabel'])
        # if 'ylabel' in plot: ax.set_ylabel(plot['ylabel'])
        ax.set_xlabel(plot.get('xlabel'))
        ax.set_ylabel(plot.get('ylabel'))
        ax.set_xlim(plot.get('xmin'), plot.get('xmax'))
        ax.set_ylim(plot.get('ymin'), plot.get('ymax'))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path)
        plt.close(fig)
    
    @classmethod
    def plot_stats(self, plot_cnf: Path) -> None:
        assert plot_cnf.is_file(), 'Plot config file does not exist.'
        with open(plot_cnf, 'r') as f:
            plot = yaml.load(f, Loader)
        plot_path = Path(plot.get('log_dir')) / plot.get('plot_dir') / f"{plot.get('name')}.png"
        fig, ax = plt.subplots()
        stats_sum = { metric: defaultdict(int) for metric in plot['metrics'] }
        for el in plot['data']:
            exp_paths = Path(el['log_dir']).glob('experiment_*')
            for exp_path in exp_paths:
                n_exp = len(list(exp_paths))
                metric_file = exp_path / Logger.STATS_FILE
                df = pd.read_csv(metric_file)
                df = df[plot['metrics']].sum(axis=0, skipna=True)
                for metric in plot['metrics']:
                    stats_sum[metric][el['label']] += df[metric] / n_exp

        print(stats_sum)
        y_bottom = [0] * len(plot['data'])
        for metric, data in stats_sum.items():
            ax.bar(data.keys(), data.values(), label=metric, bottom=y_bottom)
            y_bottom = list(data.values())

        ax.legend()
        if plot.get('grid') == True: ax.grid()
        ax.set_xlabel(plot.get('xlabel'))
        ax.set_ylabel(plot.get('ylabel'))
        ax.set_xlim(plot.get('xmin'), plot.get('xmax'))
        ax.set_ylim(plot.get('ymin'), plot.get('ymax'))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path)
        plt.close(fig)
    
    @classmethod
    def plot_times(self, plot_cnf: Path) -> None:
        assert plot_cnf.is_file(), 'Plot config file does not exist.'
        with open(plot_cnf, 'r') as f:
            plot = yaml.load(f, Loader)
        plot_path = Path(plot.get('log_dir')) / plot.get('plot_dir') / f"{plot.get('name')}.png"
        fig, ax = plt.subplots()
        stats_sum = { metric: defaultdict(int) for metric in plot['metrics'] }
        for el in plot['data']:
            exp_paths = list(Path(el['log_dir']).glob('experiment_*'))
            n_exp = len(exp_paths)
            for exp_path in exp_paths:
                logger = Logger(base_path=el['log_dir'], exp_name=exp_path.name)
                for epoch in range(1, logger.num_epochs):
                    for metric in plot['metrics']:
                        if logger.times_exist(epoch, metric):
                            tt = logger.get_times(epoch, metric).sum()[0]
                        else: 
                            tt = 0.0
                        stats_sum[metric][el['label']] += tt / n_exp / logger.num_epochs
                        # sys.exit(0)

        y_bottom = [0] * len(plot['data'])
        for metric, data in stats_sum.items():
            ax.bar(data.keys(), data.values(), label=metric, bottom=y_bottom)
            y_bottom = [ a+b for a, b in zip(y_bottom, data.values())]

        ax.legend()
        if plot.get('grid') == True: ax.grid()
        ax.set_xlabel(plot.get('xlabel'))
        ax.set_ylabel(plot.get('ylabel'))
        ax.set_xlim(plot.get('xmin'), plot.get('xmax'))
        ax.set_ylim(plot.get('ymin'), plot.get('ymax'))
        ax.set_title(plot.get('title'))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path)
        plt.close(fig)
    
    @classmethod
    def plot_timeframes(self, plot_cnf: Path) -> None:
        assert plot_cnf.is_file(), 'Plot config file does not exist.'
        with open(plot_cnf, 'r') as f:
            plot = yaml.load(f, Loader)
        plot_path = Path(plot.get('log_dir')) / plot.get('plot_dir') / f"{plot.get('name')}.png"
        fig, ax = plt.subplots()
        timeframes_dict = dict()
        def strpt(string):
            return time.mktime(time.strptime(string, '%Y-%m-%d %H:%M:%S'))
        for el in plot['data']:
            exp_paths = list(Path(el['log_dir']).glob('experiment_*'))
            n_exp = len(exp_paths)
            timeframes_dict[el['label']] = dict()
            for exp_path in exp_paths:
                logger = Logger(base_path=el['log_dir'], exp_name=exp_path.name)
                timeframes = logger.load_timeframes()
                timeframes[['time_from', 'time_to']] = timeframes[['time_from', 'time_to']].applymap(strpt)
                min_time = timeframes['time_from'].min()
                timeframes[['time_from', 'time_to']] = timeframes[['time_from', 'time_to']] - min_time
                training_timeframe = timeframes.loc[timeframes['key'].str.contains('training')]
                validation_timeframe = timeframes.loc[timeframes['key'].str.contains('validation')]
                n_val = validation_timeframe.count()[0]

                if not 'training_from' in timeframes_dict[el['label']]: timeframes_dict[el['label']]['training_from'] = 0
                if not 'training_to' in timeframes_dict[el['label']]: timeframes_dict[el['label']]['training_to'] = 0
                
                training_from = training_timeframe['time_from'].mean()
                training_to = training_timeframe['time_to'].mean()
                timeframes_dict[el['label']]['training_from'] += training_from / n_exp
                timeframes_dict[el['label']]['training_to'] += training_to / n_exp
                
                if n_val > 0:
                    if not 'validation_from' in timeframes_dict[el['label']]: timeframes_dict[el['label']]['validation_from'] = 0
                    if not 'validation_to' in timeframes_dict[el['label']]: timeframes_dict[el['label']]['validation_to'] = 0

                    validation_from = validation_timeframe['time_from'].mean()
                    validation_to = validation_timeframe['time_to'].mean()
                    timeframes_dict[el['label']]['validation_from'] += validation_from / n_exp
                    timeframes_dict[el['label']]['validation_to'] += validation_to / n_exp        

        bar_width = plot.get('bar_width', 0.2)
        bar_space = plot.get('bar_space', 0.1)
        label_train = plot.get('label_train', 'training')
        label_valid = plot.get('label_valid', 'validation')
        label_synct = plot.get('label_synct', 'train & valid')
        color_train = plot.get('color_train', 'lightsteelblue')
        color_valid = plot.get('color_valid', 'salmon')
        color_synct = plot.get('colog_synct', 'peachpuff')

        x = 0
        ticks_loc = []
        ticks_lab = []
        first_train = True
        first_valid = True
        first_synct = True

        for i, (k, v) in enumerate(timeframes_dict.items()):
            plt_training = 'training_from' in v and 'training_to' in v
            plt_validation = 'validation_from' in v and 'validation_to' in v
            plt_synchronic = plt_training and not plt_validation
            d = bar_width/2 if plt_training and plt_validation else 0
            x += bar_space + bar_width/2 + d
            ticks_loc.append(x)
            ticks_lab.append(k)
            if plt_synchronic:
                ax.barh(x+d, v['training_to'], left=v['training_from'], height=bar_width, color=color_synct, align='center', label=label_synct)
                if first_synct:
                    label_synct = None
                    first_synct = False
            elif plt_training:
                ax.barh(x+d, v['training_to'], left=v['training_from'], height=bar_width, color=color_train, align='center', label=label_train)
                if first_train:
                    label_train = None
                    first_train = False
            if plt_validation:
                ax.barh(x-d, v['validation_to'], left=v['validation_from'], height=bar_width, color=color_valid, align='center', label=label_valid)
                if first_valid:
                    label_valid = None
                    first_valid = False
            x += bar_width/2 + d
        
        ax.set_yticks(ticks_loc)
        ax.set_yticklabels(ticks_lab)

        ax.legend()
        if plot.get('grid') == True: ax.grid()
        ax.set_xlabel(plot.get('xlabel'))
        ax.set_ylabel(plot.get('ylabel'))
        ax.set_xlim(plot.get('xmin'), plot.get('xmax'))
        ax.set_ylim(plot.get('ymin'), plot.get('ymax'))
        ax.set_title(plot.get('title'))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path)
        plt.close(fig)
    
    def clear_metrics(self):
        self.logger.clear_metrics()

    def clear_plots(self):
        self.logger.clear_plots()

    def _calc_precision(self, TP: int, FP: int) -> Optional[float]:
        if not TP == 0: return TP / (TP + FP)

    def _calc_recall(self, TP: int, FN: int) -> Optional[float]:
        if not TP == 0: return TP / (TP + FN)
    
    def _calc_f1_score(self, precision: Optional[float], recall: Optional[float]) -> Optional[float]:
        if precision and recall and not precision == recall == 0: 
            return 2. * precision * recall / (precision + recall)

    def _calc_accuracy(self, TP: int, TN: int, FP: int, FN: int) -> Optional[float]:
        if not TP == TN == 0: return (TP + TN) / (TP + TN + FP + FN)
    