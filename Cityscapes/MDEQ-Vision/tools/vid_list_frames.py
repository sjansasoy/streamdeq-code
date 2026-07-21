import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Subsample Cityscapes video list files from a 20-frame list')
    parser.add_argument('mode', type=str, help='split name (e.g. val)')
    parser.add_argument(
        'num_frames',
        type=int,
        help='temporal lookback (N in val_sequence_N_frame.lst)')
    return parser.parse_args()


def main():
    args = parse_args()
    check_list = np.arange(20 - args.num_frames - 1, 20)
    f = open(f'{args.mode}_sequence_{args.num_frames}_frame.lst', 'a')
    ref_file = open(f'{args.mode}_sequence_20_frame.lst', 'r')
