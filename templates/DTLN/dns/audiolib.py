# -*- coding: utf-8 -*-
"""
@author: chkarada
"""
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import subprocess
import glob
import tqdm
import librosa
from scipy import signal
from numpy import ndarray
import matplotlib.pyplot as plt
from samplerate import Resampler

EPS = np.finfo(float).eps
np.random.seed(0)


def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def normalize(audio, target_level=-25, peak_safe=False):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    if peak_safe:
        peak = np.max(np.abs(audio))
        if scalar * peak > 1.0:
            scalar = 1.0 / peak

    audio = audio * scalar
    return audio


def normalize_segmental_rms(audio, rms, target_level=-25):
    '''Normalize the signal to the target level
    based on segmental RMS'''
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def gain(audio, db):
    '''信号增益调节'''
    amp = 10 ** (db / 20)
    audio = audio * amp
    return audio


def set_peak(audio, target_level=-25, min_level=-45):
    '''信号峰值调节'''
    peak = np.max(np.abs(audio))
    peak_level = 20 * np.log10(peak + EPS)
    # 信号过低时停止不调整增益, 避免抬升后数据异常
    if peak_level < min_level:
        # print('WARNING: Audio peak level to low. skip adjust')
        return audio
    scalar = 10 ** (target_level / 20) / (peak + EPS)
    audio = audio * scalar
    return audio


def audioread(path, norm=False, start=0, stop=None, target_level=-25):
    '''Function to read audio'''

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')
        return (None, None)

    if len(audio.shape) == 1:  # mono
        if norm:
            rms = (audio ** 2).mean() ** 0.5
            scalar = 10 ** (target_level / 20) / (rms + EPS)
            audio = audio * scalar
    else:  # multi-channel
        audio = audio.T
        audio = audio.sum(axis=0) / audio.shape[0]
        if norm:
            audio = normalize(audio, target_level)

    return audio, sample_rate


def wav_read(filename, tgt_fs=None):
    y, fs = sf.read(filename, dtype='float32')
    if tgt_fs is not None:
        if fs != tgt_fs:
            y = librosa.resample(y, orig_sr=fs, target_sr=tgt_fs)
            fs = tgt_fs
    return y, fs


def audiowrite(destpath, audio, sample_rate=16000, norm=False, target_level=-25,
               clipping_threshold=0.99, clip_test=False):
    '''Function to write audio'''

    if clip_test:
        if is_clipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError("Clipping detected in audiowrite()! " + destpath + " file not written to disk.")

    if norm:
        audio = normalize(audio, target_level)
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio / max_amp * (clipping_threshold - EPS)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return


def wav_write(data, fs, filename):
    max_value_int16 = (1 << 15) - 1
    data *= max_value_int16
    sf.write(filename, data.astype(np.int16), fs, subtype='PCM_16', format='WAV')


def add_reverb(sasxExe, input_wav, filter_file, output_wav):
    ''' Function to add reverb'''
    command_sasx_apply_reverb = "{0} -r {1} \
        -f {2} -o {3}".format(sasxExe, input_wav, filter_file, output_wav)

    subprocess.call(command_sasx_apply_reverb)
    return output_wav


def add_pyreverb(clean_speech, rir):
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0: clean_speech.shape[0]]

    return reverb_speech


def add_clipping(audio, max_thresh_perc=0.8):
    '''Function to add clipping'''
    threshold = max(abs(audio)) * max_thresh_perc
    audioclipped = np.clip(audio, -threshold, threshold)
    return audioclipped


def adsp_filter(Adspvqe, nearEndInput, nearEndOutput, farEndInput):

    command_adsp_clean = "{0} --breakOnErrors 0 --sampleRate 16000 --useEchoCancellation 0 \
                    --operatingMode 2 --useDigitalAgcNearend 0 --useDigitalAgcFarend 0 \
                    --useVirtualAGC 0 --useComfortNoiseGenerator 0 --useAnalogAutomaticGainControl 0 \
                    --useNoiseReduction 0 --loopbackInputFile {1} --farEndInputFile {2} \
                    --nearEndInputFile {3} --nearEndOutputFile {4}".format(Adspvqe,
                                                                           farEndInput, farEndInput, nearEndInput, nearEndOutput)
    subprocess.call(command_adsp_clean)


def snr_mixer(params, clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various SNR levels'''
    # cfg = params['cfg']
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))

    # Normalizing to -25 dB FS
    clean = clean / (max(abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = (clean**2).mean()**0.5

    noise = noise / (max(abs(noise)) + EPS)
    noise = normalize(noise, target_level)
    rmsnoise = (noise**2).mean()**0.5

    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = (noisyspeech**2).mean()**0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))

        scalarnoisy = scalarnoisy / noisyspeech_maxamplevel

    return clean, noisenewlevel, noisyspeech, noisy_rms_level, scalarnoisy


def segmental_snr_mixer(params, clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various segmental SNR levels'''
    # cfg = params['cfg']
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))
    clean = clean / (max(abs(clean)) + EPS)
    noise = noise / (max(abs(noise)) + EPS)
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
    noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = (noisyspeech**2).mean()**0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))

        scalarnoisy = scalarnoisy / noisyspeech_maxamplevel

    return clean, noisenewlevel, noisyspeech, noisy_rms_level, scalarnoisy


def active_rms(clean, noise, fs=16000, energy_thresh=-50):
    '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
    window_size = 100  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = (noise_win**2).mean()**0.5
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs = np.append(noise_active_segs, noise_win)
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples

    if len(noise_active_segs) != 0:
        noise_rms = (noise_active_segs**2).mean()**0.5
    else:
        noise_rms = EPS

    if len(clean_active_segs) != 0:
        clean_rms = (clean_active_segs**2).mean()**0.5
    else:
        clean_rms = EPS

    return clean_rms, noise_rms


def add_noise(audio):
    mu, sigma = 0, 0.00975  # 分布的均值（中心）, 分布的标准差（宽度）
    noise = np.random.normal(mu, sigma, audio.shape)
    audio += noise
    return audio


def activitydetector(audio, fs=16000, energy_thresh=0.13, target_level=-25):
    '''Return the percentage of the time the audio signal is above an energy threshold'''

    audio = normalize(audio, target_level,peak_safe=True)
    window_size = 50  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win**2) + EPS)
        frame_energy_prob = 1. / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)

        if smoothed_energy_prob > energy_thresh:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def resampler(input_dir, target_sr=16000, ext='*.wav'):
    '''Resamples the audio files in input_dir to target_sr'''
    files = glob.glob(f"{input_dir}/" + ext)
    for pathname in files:
        print(pathname)
        try:
            audio, fs = audioread(pathname)
            audio_resampled = librosa.core.resample(audio, fs, target_sr)
            audiowrite(pathname, audio_resampled, target_sr)
        except Exception as e:
            print(f'WARNING: resampler error. {e}')
            continue


def audio_segmenter(input_dir, dest_dir, segment_len=10, ext='*.wav'):
    '''Segments the audio clips in dir to segment_len in secs'''
    files = glob.glob(f"{input_dir}/" + ext)
    for i in range(len(files)):
        audio, fs = audioread(files[i])

        if len(audio) > (segment_len * fs) and len(audio) % (segment_len * fs) != 0:
            audio = np.append(audio, audio[0: segment_len * fs - (len(audio) % (segment_len * fs))])
        if len(audio) < (segment_len * fs):
            while len(audio) < (segment_len * fs):
                audio = np.append(audio, audio)
            audio = audio[:segment_len * fs]

        num_segments = int(len(audio) / (segment_len * fs))
        audio_segments = np.split(audio, num_segments)

        basefilename = os.path.basename(files[i])
        basename, ext = os.path.splitext(basefilename)

        for j in range(len(audio_segments)):
            newname = basename + '_' + str(j) + ext
            destpath = os.path.join(dest_dir, newname)
            audiowrite(destpath, audio_segments[j], fs)


def plot_audio(track_1, track_2, show=False, save_path=None):
    '''
    显示双轨道音频波形图
    '''
    fig, axes = plt.subplots(2, figsize=(35, 8))
    audio_len = track_1.shape[0]
    timescale = np.arange(audio_len)
    axes[0].plot(timescale, track_1)
    axes[0].set_title('track_1')
    axes[0].set_xlim([0, audio_len])

    axes[1].plot(timescale, track_2)
    axes[1].set_title('track_2')
    axes[1].set_xlim([0, audio_len])
    plt.suptitle('Waveform')
    if show:
        plt.show()
    elif save_path is not None:
        plt.savefig(save_path)


def realtime_device_task(user_obj, sample_rate=16000, block_len=512, block_shift=128):

    samplerate = 48000  # 声卡固定48K采样率
    blocksize = block_shift

    if sample_rate != samplerate:
        ratio_in = sample_rate / samplerate
        ratio_out = samplerate / sample_rate

        blocksize = int(block_shift * ratio_out)

        resampler_in = Resampler()
        resampler_out = Resampler()
        resampler_in.set_ratio(ratio_in)
        resampler_out.set_ratio(ratio_out)

    # alsa callback
    def callback(indata: ndarray, outdata: ndarray, frames: int,
                 time, status):
        if status:
            print(status)

        if sample_rate != samplerate:
            # 输入采样
            data_in = resampler_in.process(indata, ratio_in)

            # 满足一帧时调用外部处理
            if data_in.shape[0] == block_shift:
                data_out = user_obj.process(np.squeeze(data_in))
            else:
                print(f'WARNING: data_in size. {data_in.shape[0]} != {block_shift}')
                data_out = np.squeeze(data_in)

            # 输出采样
            data_out = resampler_out.process(data_out, ratio_out)

            # 输出
            if data_out.shape[0] == blocksize:
                outdata[:] = np.expand_dims(data_out, axis=-1)
            else:
                print(f'WARNING: data_out size. {data_out.shape[0]} != {blocksize}')
                outdata[:] = 0
        else:
            # 调用外部处理
            data_out = user_obj.process(np.squeeze(indata))
            # 播放
            outdata[:] = np.expand_dims(data_out, axis=-1)

    # check sound card info
    devices_list = ['Scarlett 2i2 USB', 'Scarlett Solo USB']
    for device in devices_list:
        try:
            usb_card = sd.query_devices(device=device)
        except Exception as e:
            print(f'WARNING: device. {e}')
            pass
    if 'usb_card' not in dir():
        print('==> Sound device not found!')
        exit(0)
    print('==> Sound device found: ' + usb_card['name'])

    input_device = usb_card['index']
    output_device = usb_card['index']

    # start
    try:
        with sd.Stream(samplerate=samplerate,
                       blocksize=blocksize,
                       device=(input_device, output_device),
                       channels=1,
                       dtype=np.float32,
                       latency=(usb_card['default_low_input_latency'], usb_card['default_low_output_latency']),
                       callback=callback):
            print('#' * 80)
            print('press Return to quit')
            print('#' * 80)
            input()
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        exit(e)


def realtime_wav_task(user_obj, wav_in, wav_out, sample_rate=16000, block_len=512, block_shift=128):
    print('==> read wav from: ', wav_in)
    audio, _ = wav_read(wav_in, tgt_fs=sample_rate)
    print('==> audio len: {} secs'.format(len(audio) / sample_rate))

    # preallocate output audio
    out = np.zeros((len(audio)))
    # calculate number of blocks
    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift

    for idx in tqdm.tqdm(range(num_blocks)):
        # shift data
        indata = audio[idx * block_shift:(idx * block_shift) + block_shift]
        # user process
        data_out = user_obj.process(indata)
        # save out
        out[idx * block_shift:(idx * block_shift) + block_shift] = data_out

    print('==> save wav to: ', wav_out)
    wav_write(out, sample_rate, wav_out)
