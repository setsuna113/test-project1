
import os
# import pathlib


import numpy as np
import pandas as pd
import soundfile as sf
import tqdm
import math
import multiprocessing
from pandas import DataFrame
from speechbrain.expand.DNS_Challenge.audiolib import normalize, set_peak


EPS = np.finfo(float).eps
PROCESSES = multiprocessing.cpu_count()

curr_path = os.path.dirname(os.path.abspath(__file__))


def fade_gain(audio, fade_level, max_level=0, min_level=-45, fs=16000):
    """
    整段音频淡入淡出指定dB量
    根据fade_level正负号决定淡入还是淡出
    """
    # 数据要达到1.5秒
    if len(audio) / fs < 1.5:
        return audio

    # 当前最大增益
    peak = np.max(np.abs(audio))
    peak_dBFS = (20 * np.log10(peak + EPS))
    # 淡入最大
    target_dBFS = peak_dBFS + fade_level
    # 限定最大最小范围
    target_dBFS = np.min([target_dBFS, max_level])
    target_dBFS = np.max([target_dBFS, min_level])

    # 转成线性值
    target_peak = 10 ** (target_dBFS / 20)
    # 目标电平不满足, 直接返回
    if fade_level > 0 and peak > target_peak:
        return audio
    if fade_level < 0 and peak < target_peak:
        return audio
    # 计算倍数
    scalar = target_peak / (peak + EPS)

    # 增益序列
    #   淡入时： 1.0 --->1.1-->1.2...
    #   淡出时： 1.0 --->0.9-->0.8...
    gain = np.linspace(1.0, scalar, len(audio))

    audio = audio * gain
    return audio


def generate_random_gain_waveforms(audio, target_level=-25, target_level_lower=-35, target_level_upper=-15, shift_len=3000, threshold=-50):
    '''
    解析语音静音区段
    '''
    def group_consecutive(a):
        """连续递增分离函数"""
        return np.split(a, np.where(np.diff(a) != 1)[0] + 1)

    def moving_average(interval, windowsize):
        window = np.ones(int(windowsize)) / float(windowsize)
        re = np.convolve(interval, window, 'same')
        return re

    def calculate_one_idxs(s, shift_len=3000, threshold=-50, eps=1e-7):
        """依据信号的电平计算语句分割位置"""

        # 提取峰值(太多噪声, 不用)
        # xh = signal.hilbert(s)
        # xe = np.abs(xh)

        # 使用滤波器平滑曲线
        y = np.abs(s)
        # y = savgol_filter(y, 17, 15, mode= 'nearest')
        y_av = moving_average(y, shift_len)  # 采用平均值方式更合理

        # 计算dBFS并转成数字开关量
        dBFS = 20 * np.log10(abs(y_av) + eps)
        digital_db = np.where(dBFS > threshold, 0, 1)

        # 分离出1的区间范围
        nonzero_idx = np.nonzero(digital_db.astype(int))[0]
        one_idxs = []
        start_idx = 0
        if len(nonzero_idx) != 0:
            split_labels = group_consecutive(nonzero_idx)
            for split_label in split_labels:
                if split_label[-1] - split_label[0] > shift_len:
                    zero_point = (split_label[0] + split_label[-1]) // 2
                    if start_idx != zero_point:
                        one_idxs.append([start_idx, zero_point])
                        start_idx = zero_point
        if start_idx != (len(s) - 1):
            one_idxs.append([start_idx, len(s) - 1])

        return one_idxs

    # 调内部函数提取语句间隔点
    one_idxs = calculate_one_idxs(audio, shift_len, threshold)

    # 按区间改变gain值, 作为噪声信号
    noisy_s = np.zeros(len(audio), dtype=np.float32)
    clean_s = np.zeros(len(audio), dtype=np.float32)

    # 随机数范围
    random_x = np.arange(target_level_lower, target_level_upper)
    # 概率规则
    random_rate = np.abs(random_x) / np.sum(np.abs(random_x))
    if False:

        # white_noise_s = sf.read(os.path.join(
        #     curr_path, 'data/WhiteNoise_Full_-50_dBFS_48k_Float.wav'), start=0, stop=len(audio))[0]
        # 经过对环境噪声频谱分析发现，和粉噪很类似(使用REW软件生成不同频率范围和增益范围的粉色噪声，用于训练不抬高噪声，或者去稳态噪声都是可以的)
        pink_noise_s = sf.read(os.path.join(curr_path, 'data/Office-REC-1.wav'), start=0, stop=len(audio))[0]
        pink_noise_radio = np.random.choice(a=np.arange(0.1, 1.5, 0.005))
        # 取随机目标峰值电平
        peak_level = np.random.choice(a=random_x, size=1, replace=False, p=random_rate)[0]
        # 调整到目标峰值电平
        noisy_s[:] = set_peak(audio, target_level=peak_level) + (pink_noise_s * pink_noise_radio)
        clean_s[:] = set_peak(audio, target_level=peak_level)  # +(pink_noise_s*pink_noise_radio)
    else:
        min_peak_level = 0
        for i, idxs in enumerate(one_idxs):
            # 实测网络音频在-30下, 距离从10CM-100CM-200CM的信号变化(-27, -35, -40)
            # 实测网络音频在-40下, 距离从10CM-100CM-200CM的信号变化(-17, -22, -25)
            # 取随机目标峰值电平
            peak_level = np.random.choice(a=random_x, size=1, replace=False, p=random_rate)[0]
            # rms_level = np.random.randint(target_level_lower, target_level_upper)    # (-40, -15)
            noisy_s[idxs[0]:idxs[1]] = set_peak(audio[idxs[0]:idxs[1]], target_level=peak_level)
            clean_s[idxs[0]:idxs[1]] = set_peak(audio[idxs[0]:idxs[1]], target_level=target_level)

            # 使用淡入淡出，实现noisy数据短时间内产生变化
            fade_level = np.random.choice(a=np.arange(-20, 21, 0.5))
            noisy_s[idxs[0]:idxs[1]] = fade_gain(noisy_s[idxs[0]:idxs[1]],
                                                 fade_level=fade_level,
                                                 max_level=target_level_upper,
                                                 min_level=target_level_lower)
            # gain = np.random.uniform(0.05, 1.5)
            # noisy_s[idxs[0]:idxs[1]] = audio[idxs[0]:idxs[1]] * gain
            # 更新最小峰值
            if min_peak_level > peak_level:
                min_peak_level = peak_level

        # 添加一个白噪声，使用REW-->Generator->Noise->White random->Custom(10~30, BU2, -12dB)生成类似于真实办公环境下的噪声, 这一步需使用频谱分析确认
        # pink_noise_s = sf.read(os.path.join(curr_path, 'data/PinkNoise_Full_-12_dBFS_48k_Float.wav'), start=0, stop=len(audio))[0]
        # pink_noise_level = np.random.choice(a=np.arange(-60, -45, 0.1))
        # pink_noise_adjust_s = set_peak(pink_noise_s, target_level=np.min([min_peak_level, pink_noise_level]))
        # noisy_s += pink_noise_adjust_s
        # clean_s += pink_noise_adjust_s
        # noisy_s += pink_noise_adjust_s
        # noisy_s = add_noise(noisy_s, min_peak_level)

    return noisy_s, clean_s


def create_noisys(df_list: DataFrame, level_arg):
    '''
    创建clean, niosy接口
    '''

    for item in tqdm.tqdm(df_list.itertuples(), total=df_list.shape[0]):
        audio_s, sr = sf.read(item.clean_path)
        if sr != level_arg['fs']:
            continue
        audio_s = normalize(audio_s, target_level=level_arg['target_level'], peak_safe=True)  # 原始数据先归一化一次
        noisy_s, clean_s = generate_random_gain_waveforms(audio_s,
                                                          target_level=level_arg['target_level'],  # 仿照噪声再次分段执行归一化
                                                          target_level_lower=level_arg['target_level_lower'], target_level_upper=level_arg['target_level_upper'],
                                                          shift_len=level_arg['shift_len'], threshold=level_arg['threshold'])

        # 模拟音频处理器, 对所有输入信号减20dB
        # clean_s = gain(clean_s, level_arg['clean_sub_level'])  # 因归一化了一次, 再次减-10dB比较接近实际使用电平(鹅颈麦接声卡输入旋钮在2点钟方向的信号大小相当)
        # noisy_s = gain(noisy_s, level_arg['noisy_sub_level'])

        # 显示音频波形
        # plot_audio(audio, noisy_s)

        # save
        sf.write(item.noisy_path, noisy_s, level_arg['fs'])
        sf.write(item.clean_path, clean_s, level_arg['fs'])


def agc_synthetic(noisy_speech, agc_speech):
    # 转成音频帧长
    agc_speech['shift_len'] = agc_speech['shift_len'] * agc_speech['fs']
    # 提取train.lst文件列表, 转成csv格式方便操作
    df = pd.read_csv(noisy_speech['train_list_file'], sep='\\s+', header=None)
    df.columns = ['clean_path', 'noise_path', 'noisy_path', 'audio_length', 'snr']

    total_data = df.shape[0]
    if total_data == 0:
        raise ValueError('train list is empty')

    # 分块
    if total_data < PROCESSES:
        step = total_data
    else:
        step = math.ceil(total_data / PROCESSES)
    item_idxs = np.arange(total_data)
    split_item_idxs = [item_idxs[i:i + step] for i in range(0, total_data, step)]

    # 分核执行
    pool = multiprocessing.Pool(processes=PROCESSES)
    for item_idxs in split_item_idxs:
        sub_df = df.iloc[item_idxs]
        # create_noisys(sub_df, agc_speech)  # 单任务运行
        pool.apply_async(create_noisys, (sub_df, agc_speech))

    # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    pool.close()
    pool.join()
