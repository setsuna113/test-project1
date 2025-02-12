# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:28:41 2019

@author: rocheng
"""
import os
import csv
from shutil import copyfile
import glob
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

def datasets_untar_path():
    datasets = '/home/jason/data'
    if not os.path.isdir(datasets):
        raise FileExistsError('ENV NEXUS_LOCAL_PATH/datasets_untar')
    return datasets

def datasets_archive_path():
    return os.path.expanduser('~/share/NEXUS_LOCAL/datasets_archive')


def get_dir(cfg, param_name):
    """Helper function to retrieve directory name if it exists,
       create it if it doesn't exist"""

    if param_name in cfg:
        dir_name = cfg[param_name]
    else:
        raise ValueError(f'{param_name} no define in param')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def get_file(cfg, param_name):
    """Helper function to retrieve directory name if it exists,
       create it if it doesn't exist"""

    if param_name in cfg:
        file_name = cfg[param_name]
    else:
        raise ValueError(f'{param_name} no define in param')
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return file_name


def write_log_file(log_dir, log_filename, data):
    """Helper function to write log file"""
    with open(os.path.join(log_dir, log_filename), mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            csvwriter.writerow([row])


def str2bool(string):
    return string.lower() in ("yes", "true", "t", "1")


def rename_copyfile(src_path, dest_dir, prefix='', ext='*.wav'):
    srcfiles = glob.glob(f"{src_path}/" + ext)
    for i in range(len(srcfiles)):
        dest_path = os.path.join(dest_dir, prefix + '_' + os.path.basename(srcfiles[i]))
        copyfile(srcfiles[i], dest_path)
