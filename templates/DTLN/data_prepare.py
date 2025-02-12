"""
Downloads and creates manifest files for Mini LibriSpeech.
Noise is automatically added to samples, managed by the EnvCorrupt class.

Authors:
 * Peter Plantinga, 2020
"""

import os
import logging
from random import sample
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import soundfile as sf
from dns.noisyspeech_synthesizer_multiprocessing import main_body  # NOQA: E402

logger = logging.getLogger(__name__)


def basename(path):
    filename, ext = os.path.splitext(os.path.basename(path))
    return filename, ext


def prepare_dns_noisy(
    noisy_speech, save_csv_train, save_csv_valid, save_csv_test
):
    """
    Prepares the json files for the Mini Librispeech dataset.

    Downloads the dataset if its not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.
    save_csv_train : str
        Path where the train data specification file will be saved.
    save_csv_valid : str
        Path where the validation data specification file will be saved.
    save_csv_test : str
        Path where the test data specification file will be saved.
    save_csv_metric : str
        Path where the metric  data specification file will be saved.

    Example
    -------
    >>> data_folder = '/path/to/mini_librispeech'
    >>> prepare_mini_librispeech(data_folder, 'train.json', 'valid.json', 'test.json')
    """
    # Check if this phase is already done (if so, skip it)
    if skip(save_csv_train, save_csv_valid, save_csv_test) \
            and check_folders(noisy_speech['noisy_destination'], noisy_speech['clean_destination'], noisy_speech['noise_destination'], noisy_speech['testset_dir']):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # gen train and valid data
    if not check_folders(noisy_speech['noisy_destination'], noisy_speech['clean_destination'], noisy_speech['noise_destination']) or \
            not skip(noisy_speech['train_list_file']):
        main_body(noisy_speech)

    # 拆分train.lst 为 train.csv, valid.csv
    prepare_train_valid_test_csv(noisy_speech, save_csv_train, save_csv_valid, save_csv_test)

    # 测试集csv
    #prepare_metric_csv(noisy_speech, save_csv_metric)


def prepare_train_valid_test_csv(noisy_speech, save_csv_train, save_csv_valid, save_csv_test):
    # 提取train.lst文件列表, 转成csv格式方便操作
    df = pd.read_csv(noisy_speech['train_list_file'], sep='\\s+', header=None)
    df.columns = ['clean_path', 'noise_path', 'noisy_path', 'audio_length', 'snr']

    total_data = df.shape[0]
    df_index = [i for i in range(total_data)]  # 序列编号

    # 采用7:2:1比例划分数据, 但是校验集不超过1000个, 测试集不超过100个标准
    valid_size = int(0.2 * total_data)
    if valid_size > 2000:
        valid_size = 2000
    test_size = int(0.1 * total_data)
    if test_size > 100:
        test_size = 100
    train_size = total_data - (valid_size + test_size)

    # 按比例分割csv数据表
    np.random.seed(3)
    # 乱序抽取训练集序列
    train_ids = sample(df_index, train_size)
    # 更新序列
    df_remain_index = list(set(df_index) - set(train_ids))
    # 抽取校验集序列
    valid_ids = sample(df_remain_index, valid_size)
    # 剩余给测试集
    test_ids = list(set(df_remain_index) - set(valid_ids))

    train_ids.sort()
    valid_ids.sort()
    test_ids.sort()
    df_train = df.iloc[train_ids]
    df_val = df.iloc[valid_ids]
    df_test = df.iloc[test_ids]

    Path(os.path.dirname(save_csv_train)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(save_csv_valid)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(save_csv_test)).mkdir(parents=True, exist_ok=True)
    df_train.to_csv(save_csv_train, sep=',', index=None)
    df_val.to_csv(save_csv_valid, sep=',', index=None)
    df_test.to_csv(save_csv_test, sep=',', index=None)


def prepare_metric_csv(noisy_speech, save_csv_metric):

    # test data (with_reverb and no_reverb)
    test_noisy_files_list = []
    for speech_type in ("with_reverb", "no_reverb"):
        testset_dir = os.path.join(noisy_speech['testset_dir'], speech_type, 'noisy')
        test_noisy_files_list += librosa.util.find_files(testset_dir, ext='wav')

    # test clean
    speech_with_reverb_list = []
    speech_no_reverb_list = []
    for noisy_file_path in test_noisy_files_list:
        parent_dir = Path(noisy_file_path).parents[1].name
        noisy_filename, _ = basename(noisy_file_path)

        if parent_dir == "with_reverb":
            speech_type = "With_reverb"
        elif parent_dir == "no_reverb":
            speech_type = "No_reverb"
        elif parent_dir == "dns_2_non_english":
            speech_type = "Non_english"
        elif parent_dir == "dns_2_emotion":
            speech_type = "Emotion"
        elif parent_dir == "dns_2_singing":
            speech_type = "Singing"
        else:
            raise NotImplementedError(f"Not supported dir: {parent_dir}")

        # Find the corresponding clean speech using "parent_dir" and "file_id"
        file_id = noisy_filename.split("_")[-1]
        if parent_dir in ("dns_2_emotion", "dns_2_singing"):
            # e.g., synthetic_emotion_1792_snr19_tl-35_fileid_19 => synthetic_emotion_clean_fileid_15
            clean_filename = f"synthetic_{speech_type.lower()}_clean_fileid_{file_id}"
        elif parent_dir == "dns_2_non_english":
            # e.g., synthetic_german_collection044_14_-04_CFQQgBvv2xQ_snr8_tl-21_fileid_121 => synthetic_clean_fileid_121
            clean_filename = f"synthetic_clean_fileid_{file_id}"
        else:
            # e.g., clnsp587_Unt_WsHPhfA_snr8_tl-30_fileid_300 => clean_fileid_300
            clean_filename = f"clean_fileid_{file_id}"

        clean_file_path = noisy_file_path.replace(
            f"noisy/{noisy_filename}", f"clean/{clean_filename}"
        )
        duration = min(sf.info(noisy_file_path).duration, sf.info(clean_file_path).duration)
        speech_item = (int(file_id),
                       clean_file_path,
                       noisy_file_path,
                       speech_type,
                       duration)
        if speech_type == "With_reverb":
            speech_with_reverb_list.append(speech_item)
        else:
            speech_no_reverb_list.append(speech_item)
        # print(noisy_file_path, clean_file_path, noisy_filename, speech_type)

    # 按file_id排序后再合并
    speech_with_reverb_list = sorted(speech_with_reverb_list)
    speech_no_reverb_list = sorted(speech_no_reverb_list)
    speech_list = speech_with_reverb_list + speech_no_reverb_list

    columns = ['file_id', 'clean_path', 'noisy_path', 'speech_type', 'audio_length']
    df_test = pd.DataFrame(data=speech_list, index=None, columns=columns)

    Path(os.path.dirname(save_csv_metric)).mkdir(parents=True, exist_ok=True)
    df_test.to_csv(save_csv_metric, sep=',', index=None)


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True
