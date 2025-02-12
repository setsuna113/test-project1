
import multiprocessing
import soundfile as sf
import os
import shutil
import tqdm
import numpy as np
import librosa

EPS = np.finfo(float).eps

PROCESSES = multiprocessing.cpu_count()


def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    peak = np.max(np.abs(audio))
    rms = (audio ** 2).mean() ** 0.5
    x = 10 ** (target_level / 20)  # dBFS = 20 * np.log10(abs(x) / 1.0) -> x = 10 ** (dBFS / 20)
    scalar = x / (rms + EPS)
    if scalar * peak > 1.0:
        scalar = 1.0 / peak

    audio = audio * scalar
    return audio


def calculate_signal_level(audio, eps=1e-7):
    '''计算音频信号均值和峰值电平'''
    rms = (audio ** 2).mean() ** 0.5
    peak = np.max(np.abs(audio))
    rms_level = 20 * np.log10(abs(rms) + eps)
    peak_level = 20 * np.log10(abs(peak) + eps)

    return rms_level, peak_level


def calculate_signal_segment_level(audio, sr=16000, segment_len=3.0, eps=1e-7):
    '''分段计算音频信号均值和峰值电平'''
    segment_N = int((len(audio) / sr) / segment_len)
    segment_samples = int(sr * segment_len)
    rms_levels = []
    peak_levels = []

    # 先整体归一化到-25
    audio = normalize(audio)

    # 分段检查, 偏离归一化电平在较小范围内则为稳态噪声
    for i in range(segment_N):
        frame = audio[i * segment_samples:(i + 1) * segment_samples]
        rms = (frame ** 2).mean() ** 0.5
        peak = np.max(np.abs(frame))
        rms_level = 20 * np.log10(abs(rms) + eps)
        peak_level = 20 * np.log10(abs(peak) + eps)
        # print(i, rms_level, peak_level)
        rms_levels.append(rms_level)
        peak_levels.append(peak_level)

    rms_gap = max(rms_levels) - min(rms_levels)
    peak_gap = max(peak_levels) - min(peak_levels)

    return rms_gap, peak_gap


def calculate_noises(noise_list, noise_out_path, max_rms_level=-20, max_peak_level=-10):
    '''
    创建clean, niosy接口
    '''
    for wav in tqdm.tqdm(noise_list):
        audio_s = sf.read(wav)[0]
        audio_rms, audio_peak = calculate_signal_level(audio_s)
        # 符合要求的信号拷贝到目标路径
        if audio_rms < max_rms_level and audio_peak <= max_peak_level:
            filename = os.path.split(wav)[1]
            filename_out = str(int(abs(audio_rms))) + '-' + str(int(abs(audio_peak))) + '-' + filename
            shutil.copyfile(wav, os.path.join(noise_out_path, filename_out))


def calculate_noises_in_segment(noise_list, noise_out_path, max_rms_gap=2.0, max_peak_gap=2.0):
    '''
    创建clean, niosy接口
    '''
    segment_len = 0.3
    for wav in tqdm.tqdm(noise_list):
        # 过滤掉过小的文件
        if sf.info(wav).duration < 5:
            continue
        audio_s, sr = sf.read(wav)
        rms_gap, peak_gap = calculate_signal_segment_level(audio_s, sr=sr, segment_len=segment_len)
        # 符合要求的信号拷贝到目标路径
        if rms_gap < max_rms_gap and peak_gap < max_peak_gap:
            filename = os.path.split(wav)[1]
            shutil.copyfile(wav, os.path.join(noise_out_path, filename))


# 并行筛查噪声数据电平
def noise_data_multiprocessing(noise_list, noise_out_path, max_rms_level=-20, max_peak_level=-10):
    length = len(noise_list)  # a 的长度
    # print(length)
    part = PROCESSES - 1  # 平均 P-1 等份, 避免不能取整
    step = length // part  # 步长
    split_noise_list = [noise_list[i:i + step] for i in range(0, length, step)]

    if os.path.exists(noise_out_path):
        shutil.rmtree(noise_out_path)
    os.makedirs(noise_out_path)

    # 这里设置允许同时运行的的进程数量要考虑机器cpu的数量，进程的数量最好别小于cpu的数量，
    # 因为即使大于cpu的数量，增加了任务调度的时间，效率反而不能有效提高
    pool = multiprocessing.Pool(processes=PROCESSES)
    for flie_list in split_noise_list:
        # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        # print(clean_agc)
        # pool.apply_async(calculate_noises, (flie_list, noise_out_path, max_rms_level, max_peak_level))
        pool.apply_async(calculate_noises_in_segment, (flie_list, noise_out_path, 2.0, 2.0))

    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束


noise_path = '/home/jason/audio/recipes/ANS/DTLN/data/noise'
noise_out_path = '/home/jason/audio/recipes/ANS/DTLN/data/noise'

noise_list = librosa.util.find_files(noise_path, ext='wav')


noise_data_multiprocessing(noise_list, noise_out_path, max_rms_level=-30, max_peak_level=-25)
