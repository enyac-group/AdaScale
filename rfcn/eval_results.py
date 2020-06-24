import _init_paths
import cPickle
import argparse
import cv2
import os
import numpy as np
from dataset import *

from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate mAP')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--result', help='detection results (.txt)', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    imdb = eval(config.dataset.dataset)(config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path)
    imdb.do_python_eval_filename(args.result)

if __name__ == '__main__':
    main()