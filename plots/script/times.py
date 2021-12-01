import argparse
from pathlib import Path
from cliva_fl.utils import Plotter

parser = argparse.ArgumentParser(description='Argument parser for summary plot creation.')
parser.add_argument('-c', '--conf', type=str, required=True, help='Path to the summary plot config file.')
parser.add_argument('-t', '--trim', action='store_true', help='Trim the metric name by splitting at "_" and using the last element as label.')

args = parser.parse_args()

plot_cnf = Path(args.conf)
Plotter.plot_times(plot_cnf, trim_label=args.trim)