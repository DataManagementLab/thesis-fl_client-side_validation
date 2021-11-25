import argparse
from pathlib import Path
from cliva_fl.utils import Plotter

parser = argparse.ArgumentParser(description='Argument parser for summary plot creation.')
parser.add_argument('-c', '--conf', type=Path, required=True, help='Path to the summary plot config file.')

args = parser.parse_args()
Plotter.plot_line_quality(args.conf)
