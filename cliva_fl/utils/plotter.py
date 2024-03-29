from collections import defaultdict
from os import stat
from pathlib import Path
from numpy import imag, zeros
from torch.nn.modules.module import T
from torch.utils.data import DataLoader
from typing import List, Optional
import sys, yaml, time
from yaml import Loader
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from cliva_fl.utils import Logger
from cliva_fl import models, datasets, utils

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
    
    @staticmethod
    def load_plot_config(path: Path):
        assert path.is_file(), 'Plot config file does not exist.'
        with open(path, 'r') as f:
            return yaml.load(f, Loader)
    
    @staticmethod
    def get_plot_path(plot_cnf):
        return Path(plot_cnf.get('log_dir')) / plot_cnf.get('plot_dir') / f"{plot_cnf.get('name')}.png"
    
    @classmethod
    def plot_summary(cls, plot_cnf: Path) -> None:
        plot = cls.load_plot_config(plot_cnf)
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
        fig.savefig(plot_path, bbox_inches = "tight")
        plt.close(fig)
    
    @classmethod
    def plot_stats(cls, plot_cnf: Path) -> None:
        plot = cls.load_plot_config(plot_cnf)
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
        fig.savefig(plot_path, bbox_inches = "tight")
        plt.close(fig)
    
    @classmethod
    def plot_times(cls, plot_cnf: Path, trim_label=False) -> None:
        plot = cls.load_plot_config(plot_cnf)
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
            if trim_label:
                labl = metric.split('_')[-1].title()
            else:
                labl = metric
            ax.bar(data.keys(), data.values(), label=labl, bottom=y_bottom)
            y_bottom = [ a+b for a, b in zip(y_bottom, data.values())]

        ax.legend()
        if plot.get('grid') == True: ax.grid()
        ax.set_xlabel(plot.get('xlabel'))
        ax.set_ylabel(plot.get('ylabel'))
        ax.set_xlim(plot.get('xmin'), plot.get('xmax'))
        ax.set_ylim(plot.get('ymin'), plot.get('ymax'))
        ax.set_title(plot.get('title'))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, bbox_inches = "tight")
        plt.close(fig)
    
    @classmethod
    def plot_timeframes(cls, plot_cnf: Path) -> None:
        plot = cls.load_plot_config(plot_cnf)
        plot_path = Path(plot.get('log_dir')) / plot.get('plot_dir') / f"{plot.get('name')}.png"
        fig, ax = plt.subplots()
        timeframes_dict = dict()
        def strpt(string):
            return time.mktime(time.strptime(string, '%Y-%m-%d %H:%M:%S'))
        for el in plot['data']:
            exp_paths = list(Path(el['log_dir']).glob('experiment_*'))
            n_exp = len(exp_paths)
            timeframes_dict[el['label']] = dict(conf=el['conf'] if 'conf' in el else dict())
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
        color_synct = plot.get('color_synct', 'peachpuff')

        x = 0
        ticks_loc = []
        ticks_lab = []
        seen_labels = list()

        for i, (k, v) in enumerate(timeframes_dict.items()):
            plt_training = 'training_from' in v and 'training_to' in v
            plt_validation = 'validation_from' in v and 'validation_to' in v
            plt_synchronic = plt_training and not plt_validation
            d = bar_width/2 if plt_training and plt_validation else 0
            x += bar_space + bar_width/2 + d
            conf = v['conf']
            ticks_loc.append(x)
            ticks_lab.append(k)
            if plt_synchronic:
                colr = conf['color'] if 'color' in conf else color_synct
                labl = conf['type'] if 'type' in conf else label_synct
                if labl in seen_labels: labl = None
                else: seen_labels.append(labl)
                ax.barh(x+d, v['training_to'], left=v['training_from'], height=bar_width, color=colr, align='center', label=labl)
            elif plt_training:
                colr = conf['t_color'] if 't_color' in conf else color_train
                labl = conf['t_type'] if 't_type' in conf else label_train
                if labl in seen_labels: labl = None
                else: seen_labels.append(labl)
                ax.barh(x+d, v['training_to'], left=v['training_from'], height=bar_width, color=color_train, align='center', label=labl)
            if plt_validation:
                colr = conf['v_color'] if 'v_color' in conf else color_valid
                labl = conf['v_type'] if 'v_type' in conf else label_valid
                if labl in seen_labels: labl = None
                else: seen_labels.append(labl)
                ax.barh(x-d, v['validation_to'], left=v['validation_from'], height=bar_width, color=color_valid, align='center', label=labl)
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
        fig.savefig(plot_path, bbox_inches = "tight")
        plt.close(fig)
    
    @classmethod
    def plot_line_times(cls, plot_cnf: Path) -> None:
        # READ CONFIG
        plot = cls.load_plot_config(plot_cnf)
        plot_path = cls.get_plot_path(plot)

        # COLLECT DATA
        data = dict()
        for el in plot['data']:
            data[el['label']] = dict(x=plot['xvalues'], y=list(), conf=el['conf'] if 'conf' in el else dict())
            for log_dir in el['log_dirs']:
                exp_paths = list(Path(log_dir).glob('experiment_*'))
                n_exp = len(exp_paths)
                yvalue = 0.0
                for exp_path in exp_paths:
                    logger = Logger(base_path=log_dir, exp_name=exp_path.name)
                    for epoch in range(1, logger.num_epochs):
                        yvalue += logger.get_times(epoch, plot['metric']).sum()[0] / n_exp / logger.num_epochs
                data[el['label']]['y'].append(yvalue)
                # sys.exit(0)

        # CREATE PLOT
        cls.plot_linechart(data, plot, plot_path)
    
    @classmethod
    def plot_line_memory(cls, plot_cnf: Path) -> None:
        # READ CONFIG
        plot = cls.load_plot_config(plot_cnf)
        plot_path = cls.get_plot_path(plot)

        # COLLECT DATA
        data = dict()
        for el in plot['data']:
            data[el['label']] = dict(x=plot['xvalues'], y=list(), conf=el['conf'] if 'conf' in el else dict())
            for log_dir in el['log_dirs']:
                exp_paths = list(Path(log_dir).glob('experiment_*'))
                n_exp = len(exp_paths)
                yvalue = 0.0
                for exp_path in exp_paths:
                    logger = Logger(base_path=log_dir, exp_name=exp_path.name)
                    mem = logger.load_memory_usage()
                    yvalue +=  mem[plot['metric']].mean() / 1000 / 1000 / n_exp
                data[el['label']]['y'].append(yvalue)
                # sys.exit(0)

        # CREATE PLOT
        cls.plot_linechart(data, plot, plot_path)
    
    @classmethod
    def plot_line_quality(cls, plot_cnf: Path) -> None:
        # READ CONFIG
        plot = cls.load_plot_config(plot_cnf)
        plot_path = cls.get_plot_path(plot)

        # COLLECT DATA
        data = dict()
        for el in plot['data']:
            data[el['label']] = dict(x=plot['xvalues'], y=list(), conf=el['conf'] if 'conf' in el else dict())
            for log_dir in el['log_dirs']:
                exp_paths = list(Path(log_dir).glob('experiment_*'))
                n_exp = len(exp_paths)
                yvalue = 0.0
                n_metrics = 0
                for exp_path in exp_paths:
                    logger = Logger(base_path=log_dir, exp_name=exp_path.name)
                    for epoch in range(1, logger.num_epochs):
                        TP, FP, FN = logger.check_attack_detection(epoch)
                        if plot['metric'] == 'precision': 
                            metric = cls._calc_precision(TP, FP)
                        if plot['metric'] == 'recall': 
                            metric = cls._calc_recall(TP, FN)
                        if plot['metric'] == 'f1_score': 
                            metric = cls._calc_f1_score(
                                cls._calc_precision(TP, FP),
                                cls._calc_recall(TP, FN))
                        if plot['metric'] == 'accuracy': 
                            metric = cls._calc_accuracy(TP, 0, FP, FN)
                        if metric:
                            n_metrics +=1
                            yvalue += metric
                        # else:
                        #     print(metric, el['label'], log_dir, exp_path.name, epoch, TP, FP, FN)
                data[el['label']]['y'].append(yvalue/n_metrics if n_metrics else 0)
                # sys.exit(0)

        # CREATE PLOT
        cls.plot_linechart(data, plot, plot_path)
    
    @staticmethod
    def plot_linechart(data, plot, plot_path: Path):
        """
        data = dict(
            method1=dict(
                x=list(),
                y=list(),
                conf=dict()
            )
        ),
        plot = dict()
        """
        # CREATE PLOT
        fig, ax = plt.subplots()

        # FILL PLOT
        for method, d in data.items():
            ax.bar(d['x'], d['y'], label=method, **d['conf'])

        # FINISH PLOT
        ax.legend()
        if plot.get('grid') == True: ax.grid()
        ax.set_xlabel(plot.get('xlabel'))
        ax.set_ylabel(plot.get('ylabel'))
        if 'xmin' in plot: ax.set_xlim(xmin=plot['xmin'])
        if 'xmax' in plot: ax.set_xlim(xmax=plot['xmax'])
        if 'ymin' in plot: ax.set_ylim(ymin=plot['ymin'])
        if 'ymax' in plot: ax.set_ylim(ymax=plot['ymax'])
        if 'yscale' in plot: ax.set_yscale(plot['yscale'], base=plot.get('ybase', 10))
        if 'xscale' in plot: ax.set_xscale(plot['xscale'], base=plot.get('xbase', 10))
        ax.set_title(plot.get('title'))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, bbox_inches = "tight")
        plt.close(fig)
    
    @staticmethod
    def plot_barchart(data, plot, plot_path: Path, width = 0.35):
        """
        data = dict(
            method1=dict(
                x=list(),
                y=list(),
                conf=dict()
            )
        ),
        plot = dict()
        """
        # CREATE PLOT
        fig, ax = plt.subplots()

        n = len(data)
        diff = np.arange(-(n/2-0.5)*width, n*width/2, width)
        if 'xticks' in plot:
            print(plot['xticks'])
            ax.set_xticks(np.arange(len(plot['xticks'])))
            ax.set_xticklabels(plot['xticks'])
        else:
            xticks = list(data.values())[0]['x']
            ax.set_xticks(np.arange(len(xticks)))
            ax.set_xticklabels(xticks)

        # FILL PLOT
        for s, (method, d) in zip(diff, data.items()):
            x = np.arange(s, s+len(d['x']))  # the label locations
            ax.bar(x, d['y'], width, label=method, **d['conf'])

        # FINISH PLOT
        ax.legend(loc=4)
        if plot.get('grid') == True: ax.grid()
        ax.set_xlabel(plot.get('xlabel'))
        ax.set_ylabel(plot.get('ylabel'))
        if 'xmin' in plot: ax.set_xlim(xmin=plot['xmin'])
        if 'xmax' in plot: ax.set_xlim(xmax=plot['xmax'])
        if 'ymin' in plot: ax.set_ylim(ymin=plot['ymin'])
        if 'ymax' in plot: ax.set_ylim(ymax=plot['ymax'])
        if 'yscale' in plot: ax.set_yscale(plot['yscale'], base=plot.get('ybase', 10))
        if 'xscale' in plot: ax.set_xscale(plot['xscale'], base=plot.get('xbase', 10))
        ax.set_title(plot.get('title'))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, bbox_inches = "tight")
        plt.close(fig)

    
    def clear_metrics(self):
        self.logger.clear_metrics()

    def clear_plots(self):
        self.logger.clear_plots()

    @staticmethod
    def _calc_precision(TP: int, FP: int) -> Optional[float]:
        if not TP == 0: return TP / (TP + FP)

    @staticmethod
    def _calc_recall(TP: int, FN: int) -> Optional[float]:
        if not TP == 0: return TP / (TP + FN)

    @staticmethod
    def _calc_f1_score(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
        if precision and recall and not precision == recall == 0: 
            return 2. * precision * recall / (precision + recall)

    @staticmethod
    def _calc_accuracy(TP: int, TN: int, FP: int, FN: int) -> Optional[float]:
        if not TP == TN == 0: return (TP + TN) / (TP + TN + FP + FN)
    