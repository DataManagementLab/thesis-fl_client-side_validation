import sys, argparse
from cliva_fl.utils import Logger, Plotter
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser(description='Argument parser for log processing and plot creation')
parser.add_argument('-n', '--num', type=int, required=False, help='Index of previously executed experiment to select for processing. Last experiment equals --num 1.')
parser.add_argument('-d', '--dir', type=str, required=False, help='Name of experiment directory to select for processing.')
parser.add_argument('-l', '--log_dir', type=str, required=False, default='logs', help='Log directory to select for processing.')
parser.add_argument('-g', '--generate', action='store_true', required=False, help='Flag to indicate weather metrics should be (re)generated from logs.')
parser.add_argument('-c', '--clean', action='store_true', required=False, help='Flag to indicate a full clean-up of all generated files.')
parser.add_argument('-m', '--metrics', nargs='+', required=True, help='List of metrics to be computed. Available metrics are {}'.format(Plotter.METRICS))

parser.add_argument('--xmin', type=float, default=None, required=False, help='Minimum value of x-axes in plot.')
parser.add_argument('--xmax', type=float, default=None, required=False, help='Maximum value of x-axes in plot.')
parser.add_argument('--ymin', type=float, default=None, required=False, help='Minimum value of y-axes in plot.')
parser.add_argument('--ymax', type=float, default=None, required=False, help='Maximum value of y-axes in plot.')

# parser.add_argument('-p', '--plot_type', type=str, default=None, required=False, help='Plot type for the plot.')

args = parser.parse_args()

assert args.num or args.dir, 'You are required to specify num or dir parameter.'
assert not (args.num and args.dir), 'You can not use num and dir parameter simultaneously to select a log directory'

if args.num:
    exp_dirs = sorted(list(Path(args.log_dir).glob('experiment_*')))
    assert args.num <= len(exp_dirs), 'Num can not be larger than the number of existing experiment directories'
    p = exp_dirs[-args.num].name
elif args.dir:
    p = Path(args.dir).name

_, YEAR, MONTH, DAY, _, TIME = p.split('_')
HOUR, MINUTE = TIME.split(':')

print(f'Experiment: {p}\nDate: {DAY}.{MONTH}.{YEAR}\tTime: {HOUR}:{MINUTE}')

timestamp = datetime(year=int(YEAR), month=int(MONTH), day=int(DAY), hour=int(HOUR), minute=int(MINUTE))
logger = Logger(base_path=args.log_dir, timestamp=timestamp)

plotter = Plotter(logger)

if args.clean:
    print('Cleaning up all generated files.')
    plotter.clear_metrics()
    plotter.clear_plots()

if args.generate:
    print('Generating metrics from logs.')
    plotter.clear_metrics()
    plotter.generate()

for metric in args.metrics:
    assert metric in Plotter.METRICS, f'Metric {metric} is not a valid metric.'
    plotter.plot_metric(metric, args.ymin, args.ymax, args.xmin, args.xmax)