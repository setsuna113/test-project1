import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import soundfile as sf
import librosa
EPS = np.finfo(float).eps

# 集成Dataset, 重写一下方法:
#   __init__
#   __getitem__
#   __len__


class DataGen(Dataset):
    def __init__(self,
                 csv_file,
                 length_per_sample=4,
                 fs=16000,
                 mode='train'):
        self.mode = mode

        self.df = pd.read_csv(csv_file)

        self.data_len = self.df.shape[0]

        self.fs = fs
        self.length_per_sample = length_per_sample

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        batch = {}

        data_row = self.df.iloc[idx]

        audio_length = data_row['audio_length']
        # clean_path = data_row['clean_path']
        # position_find = clean_path.find("snr")
        # noiseFlag = clean_path[position_find+3:position_find+6]

        # read data
        clean_s, fs_clean = sf.read(data_row['clean_path'], dtype='float32')
        noisy_s, fs_noisy = sf.read(data_row['noisy_path'], dtype='float32')

        # 采样检查
        if fs_clean != self.fs:
            clean_s = librosa.resample(clean_s, orig_sr=fs_clean, target_sr=self.fs)
        if fs_noisy != self.fs:
            noisy_s = librosa.resample(noisy_s, orig_sr=fs_noisy, target_sr=self.fs)
       

        # noisy已提前合成好, 读取时候两者位置须对齐
        if self.mode == 'test' or self.mode == 'test_metric':
            clean_s = clean_s[0: int(audio_length * self.fs)]
            noisy_s = noisy_s[0: int(audio_length * self.fs)]
        else:
            Begin = int(np.random.uniform(0, audio_length - self.length_per_sample)) * self.fs
            clean_s = clean_s[Begin: int(Begin + self.length_per_sample * self.fs)]
            noisy_s = noisy_s[Begin: int(Begin + self.length_per_sample * self.fs)]

        batch['idxs'] = str(idx)
        batch['mode'] = self.mode
        batch['clean'] = torch.from_numpy(clean_s)
        batch['noisy'] = torch.from_numpy(noisy_s)
        
        if 'clean_path' in data_row:
            batch['clean_path'] = data_row['clean_path']
        if 'noisy_path' in data_row:
            batch['noisy_path'] = data_row['noisy_path']
        if 'speech_type' in data_row:
            batch['speech_type'] = data_row['speech_type']

        return batch  # 直接返回batch, 标签留住就是字典了


def train_dataloader(hparams):
    """训练集（training set）: 训练算法"""
    TrainData = DataGen(hparams["train_annotation"], hparams["length_per_sample"], hparams["sample_rate"], mode='train')
    trainloader = DataLoader(TrainData, batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["num_workers"])
    return trainloader


def valid_dataloader(hparams):
    """开发集（development set）：调整参数、选择特征，以及对学习算法作出其它决定。"""
    DevData = DataGen(hparams["valid_annotation"], hparams["length_per_sample"], hparams["sample_rate"], mode='valid')
    devloader = DataLoader(DevData, batch_size=hparams["batch_size"], shuffle=False, num_workers=hparams["num_workers"])
    return devloader


def test_dataloader(hparams):
    """测试集（test set）：开发集中选出的最优的模型在测试集上进行评估。不会据此改变学习算法或参数。"""
    EvalData = DataGen(hparams["test_annotation"], hparams["length_per_sample"], hparams["sample_rate"], mode='test')
    evalloader = DataLoader(EvalData, batch_size=1, shuffle=False, num_workers=0)
    return evalloader


def test_metric_dataloader(hparams):
    """评估集（metric set）：开发集中选出的最优的模型在测试集上进行评估。不会据此改变学习算法或参数。"""
    MetricData = DataGen(hparams["metric_annotation"], hparams["length_per_sample"], hparams["sample_rate"], mode='test_metric')
    metricloader = DataLoader(MetricData, batch_size=1, shuffle=False, num_workers=0)
    return metricloader
