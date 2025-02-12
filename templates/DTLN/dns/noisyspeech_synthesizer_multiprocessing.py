"""
@author: chkarada
"""

# Note that this file picks the clean speech files randomly, so it does not guarantee that all
# source files will be used


import os
import glob
from itertools import repeat
import multiprocessing
import random
from random import shuffle
import librosa
import numpy as np
import re
import pandas as pd
from scipy.io import wavfile

from dns.audiolib import is_clipped, audioread, audiowrite, snr_mixer, activitydetector, add_pyreverb
from dns.utils import get_dir, get_file, write_log_file

PROCESSES = multiprocessing.cpu_count()
MAXTRIES = 50
MAXFILELEN = 100

np.random.seed(2)
random.seed(3)

clean_counter = None
noise_counter = None


def init(args1, args2):
    ''' store the counter for later use '''
    global clean_counter, noise_counter
    clean_counter = args1
    noise_counter = args2


def build_audio(is_clean, params, filenum, audio_samples_length=-1):
    '''Construct an audio signal from source files'''

    fs_output = params['fs']
    silence_length = params['silence_length']
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length'] * params['fs'])

    output_audio = np.zeros(0)
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    global clean_counter, noise_counter
    if is_clean:
        source_files = params['cleanfilenames']
        idx_counter = clean_counter
    else:
        source_files = params['noisefilenames']
        idx_counter = noise_counter

    # initialize silence
    silence = np.zeros(int(fs_output * silence_length))

    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    while remaining_length > 0 and tries_left > 0:

        # read next audio file and resample if necessary
        with idx_counter.get_lock():
            idx_counter.value += 1
            idx = idx_counter.value % np.size(source_files)

        input_audio, fs_input = audioread(source_files[idx])
        if fs_input != fs_output:
            input_audio = librosa.resample(input_audio, orig_sr=fs_input, target_sr=fs_output)

        # if current file is longer than remaining desired length, and this is
        # noise generation or this is training set, subsample it randomly
        if len(input_audio) > remaining_length and (not is_clean or not params['is_test_set']):
            idx_seg = np.random.randint(0, len(input_audio) - remaining_length)
            input_audio = input_audio[idx_seg:idx_seg + remaining_length]

        # check for clipping, and if found move onto next file
        if is_clipped(input_audio):
            clipped_files.append(source_files[idx])
            tries_left -= 1
            continue

        # concatenate current input audio to output audio stream
        files_used.append(source_files[idx])
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)

        # add some silence if we have not reached desired audio length
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio = np.append(output_audio, silence[:silence_len])
            remaining_length -= silence_len

    if tries_left == 0:
        print("Audio generation failed for filenum " + str(filenum))
        return [], [], clipped_files

    return output_audio, files_used, clipped_files


def gen_audio(is_clean, params, filenum, audio_samples_length=-1):
    '''Calls build_audio() to get an audio signal, and verify that it meets the
       activity threshold'''

    clipped_files = []
    low_activity_files = []
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length'] * params['fs'])
    if is_clean:
        activity_threshold = params['clean_activity_threshold']
    else:
        activity_threshold = params['noise_activity_threshold']

    while True:
        audio, source_files, new_clipped_files = \
            build_audio(is_clean, params, filenum, audio_samples_length)

        clipped_files += new_clipped_files
        if len(audio) < audio_samples_length:
            continue

        if activity_threshold == 0.0:
            break

        percactive = activitydetector(audio=audio)
        if percactive > activity_threshold:
            break
        else:
            low_activity_files += source_files

    return audio, source_files, clipped_files, low_activity_files


def main_gen(params, filenum):
    '''Calls gen_audio() to generate the audio signals, verifies that they meet
       the requirements, and writes the files to storage'''

    print("Generating file #" + str(filenum))

    clean_clipped_files = []
    clean_low_activity_files = []
    noise_clipped_files = []
    noise_low_activity_files = []
    train_list = []
    while True:
        # generate clean speech
        clean, clean_source_files, clean_cf, clean_laf = \
            gen_audio(True, params, filenum)

        # add reverb with selected RIR
        # rir_index = random.randint(0, len(params['myrir']) - 1)

        # my_rir = os.path.normpath(os.path.join(params['rir_dir'], params['myrir'][rir_index]))
        # (fs_rir, samples_rir) = wavfile.read(my_rir)

        # my_channel = int(params['mychannel'][rir_index])

        # if samples_rir.ndim == 1:
        #     samples_rir_ch = np.array(samples_rir)
        # else:
        #     samples_rir_ch = samples_rir[:, my_channel - 1]

        # if fs_rir != params['fs']:
        #     samples_rir_ch = librosa.resample(samples_rir_ch, orig_sr=fs_rir, target_sr=params['fs'])

        # if params['add_reverb_data'] == 1:
        #     reverb_rate = np.random.rand()
        #     if reverb_rate < params['percent_for_adding_reverb']:
        #         clean = add_pyreverb(clean, samples_rir_ch)  # 带混响clean

        # generate noise
        noise, noise_source_files, noise_cf, noise_laf = \
            gen_audio(False, params, filenum, len(clean))

        clean_clipped_files += clean_cf
        clean_low_activity_files += clean_laf
        noise_clipped_files += noise_cf
        noise_low_activity_files += noise_laf

        # mix clean speech and noise
        # if specified, use specified SNR value
        if not params['randomize_snr']:
            snr = params['snr']
        # use a randomly sampled SNR value between the specified bounds
        else:
            snr = np.random.randint(params['snr_lower'], params['snr_upper'])

        clean_snr, noise_snr, noisy_snr, target_level, scalar = snr_mixer(params=params,
                                                                          clean=clean,
                                                                          noise=noise,
                                                                          snr=snr)
        # Uncomment the below lines if you need segmental SNR and comment the above lines using snr_mixer
        # clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(params=params,
        #                                                                    clean=clean,
        #                                                                    noise=noise,
        #                                                                    snr=snr)
        # unexpected clipping
        if is_clipped(clean_snr) or is_clipped(noise_snr) or is_clipped(noisy_snr):
            continue
        else:
            break

    # write resultant audio streams to files
    hyphen = '-'
    clean_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in clean_source_files]
    clean_files_joined = hyphen.join(clean_source_filenamesonly)[:MAXFILELEN]
    noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in noise_source_files]
    noise_files_joined = hyphen.join(noise_source_filenamesonly)[:MAXFILELEN]

    cleanfilename = 'fileid_' + str(filenum) + '.wav'
    noisefilename = 'fileid_' + str(filenum) + '.wav'
    noisyfilename = 'fileid_' + str(filenum) + '.wav'
    # rirfilename = 'rir_fileid_' + str(filenum) + '.wav'

    cleanpath = os.path.join(params['clean_proc_dir'], cleanfilename)
    noisepath = os.path.join(params['noise_proc_dir'], noisefilename)
    noisypath = os.path.join(params['noisyspeech_dir'], noisyfilename)
    # rirpath = os.path.join(params['rir_proc_dir'], rirfilename)
    train_item = '' \
        + cleanpath + ' ' \
        + noisepath + ' ' \
        + noisypath + ' ' \
        + f'{params["audio_length"]}' + ' ' \
        + f'{snr}'
    train_list.append(train_item)

    audio_signals = [noisy_snr, clean_snr, noise_snr]
    file_paths = [noisypath, cleanpath, noisepath]

    for i in range(len(audio_signals)):
        try:
            audiowrite(file_paths[i], audio_signals[i], params['fs'])
        except Exception as e:
            print(str(e))
            pass

    return clean_source_files, clean_clipped_files, clean_low_activity_files, \
        noise_source_files, noise_clipped_files, noise_low_activity_files, train_list

def extract_list(input_list, index):
    output_list = [i[index] for i in input_list]
    flat_output_list = [item for sublist in output_list for item in sublist]
    flat_output_list = sorted(set(flat_output_list))
    return flat_output_list


def main_body(cfg):
    '''Main body of this file'''

    params = dict()
    params['cfg'] = cfg

    clean_dir = cfg['speech_dir']
    if not os.path.exists(clean_dir):
        assert False, ('Clean speech data is required')

    noise_dir = cfg['noise_dir']
    if not os.path.exists:
        assert False, ('Noise data is required')

    params['train_list_file'] = get_file(cfg, 'train_list_file')

    params['fs'] = int(cfg['sampling_rate'])
    params['audioformat'] = cfg['audioformat']
    params['audio_length'] = float(cfg['audio_length'])
    params['silence_length'] = float(cfg['silence_length'])
    params['total_hours'] = float(cfg['total_hours'])

    # clean singing speech
    params['use_singing_data'] = int(cfg['use_singing_data'])
    params['clean_singing'] = str(cfg['clean_singing'])
    params['singing_choice'] = int(cfg['singing_choice'])

    # clean emotional speech
    params['use_emotion_data'] = int(cfg['use_emotion_data'])
    params['clean_emotion'] = str(cfg['clean_emotion'])

    # clean mandarin speech
    params['use_mandarin_data'] = int(cfg['use_mandarin_data'])
    params['clean_mandarin'] = str(cfg['clean_mandarin'])

    # rir
    params['add_reverb_data'] = int(cfg['add_reverb_data'])
    params['rir_dir'] = str(cfg['rir_dir'])
    params['rir_choice'] = int(cfg['rir_choice'])
    params['lower_t60'] = float(cfg['lower_t60'])
    params['upper_t60'] = float(cfg['upper_t60'])
    params['percent_for_adding_reverb'] = float(cfg['percent_for_adding_reverb'])
    if params['fs'] == 16000:
        params['rir_table_csv'] = os.path.join(os.path.dirname(__file__), 'datasets/acoustic_params/RIR_table_simple.csv')
    elif params['fs'] == 48000:
        params['rir_table_csv'] = os.path.join(os.path.dirname(__file__), 'datasets/acoustic_params/RIR_table_simple_48k.csv')
    params['clean_speech_t60_csv'] = os.path.join(os.path.dirname(__file__), 'datasets/acoustic_params/cleanspeech_table_t60_c50.csv')

    if cfg['fileindex_start'] != 'None' and cfg['fileindex_end'] != 'None':
        params['fileindex_start'] = int(cfg['fileindex_start'])
        params['fileindex_end'] = int(cfg['fileindex_end'])
        params['num_files'] = int(params['fileindex_end']) - int(params['fileindex_start'])
    else:
        params['num_files'] = int((params['total_hours'] * 60 * 60) / params['audio_length'])
        params['fileindex_start'] = 0
        params['fileindex_end'] = params['num_files']

    print('Number of files to be synthesized:', params['num_files'])

    params['is_test_set'] = bool(cfg['is_test_set'])
    params['clean_activity_threshold'] = float(cfg['clean_activity_threshold'])
    params['noise_activity_threshold'] = float(cfg['noise_activity_threshold'])
    params['snr_lower'] = int(cfg['snr_lower'])
    params['snr_upper'] = int(cfg['snr_upper'])

    params['randomize_snr'] = bool(cfg['randomize_snr'])
    params['target_level_lower'] = int(cfg['target_level_lower'])
    params['target_level_upper'] = int(cfg['target_level_upper'])

    if 'snr' in cfg.keys():
        params['snr'] = int(cfg['snr'])
    else:
        params['snr'] = int((params['snr_lower'] + params['snr_upper']) / 2)

    params['clean_proc_dir'] = get_dir(cfg, 'clean_destination')
    params['noise_proc_dir'] = get_dir(cfg, 'noise_destination')
    params['noisyspeech_dir'] = get_dir(cfg, 'noisy_destination')
    # params['rir_proc_dir'] = get_dir(cfg, 'rir_destination')

    if 'speech_csv' in cfg.keys() and cfg['speech_csv'] != 'None':
        cleanfilenames = pd.read_csv(cfg['speech_csv'])
        cleanfilenames = cleanfilenames['filename']
    else:
        cleanfilenames = glob.glob(os.path.join(clean_dir, '**', params['audioformat']), recursive=True)

    print('clean speech:', len(cleanfilenames))
    # add singing voice to clean speech
    if params['use_singing_data'] == 1:
        if not os.path.exists(params['clean_singing']):
            assert False, ('Clean singing data is required')

        all_singing = glob.glob(os.path.join(params['clean_singing'], '**', params['audioformat']), recursive=True)

        if params['singing_choice'] == 1:  # male speakers
            mysinging = [s for s in all_singing if ("male" in s and "female" not in s)]

        elif params['singing_choice'] == 2:  # female speakers
            mysinging = [s for s in all_singing if "female" in s]

        elif params['singing_choice'] == 3:  # both male and female
            mysinging = all_singing
        else:  # default both male and female
            mysinging = all_singing

        print('Clean singing:', len(mysinging))
        if mysinging is not None:
            cleanfilenames = cleanfilenames + mysinging
    else:
        print('NOT using singing data for training!')

    # add emotion data to clean speech
    if params['use_emotion_data'] == 1:
        if not os.path.exists(params['clean_emotion']):
            assert False, ('Clean emotion data is required')

        all_emotion = glob.glob(os.path.join(params['clean_emotion'], '**', params['audioformat']), recursive=True)

        print('Clean emotion:', len(all_emotion))
        if all_emotion is not None:
            cleanfilenames = cleanfilenames + all_emotion
    else:
        print('NOT using emotion data for training!')

    # add mandarin data to clean speech
    if params['use_mandarin_data'] == 1:
        if not os.path.exists(params['clean_mandarin']):
            assert False, ('Clean mandarin data is required')

        all_mandarin = glob.glob(os.path.join(params['clean_mandarin'], '**', params['audioformat']), recursive=True)

        print('Clean mandarin:', len(all_mandarin))
        if all_mandarin is not None:
            cleanfilenames = cleanfilenames + all_mandarin
    else:
        print('NOT using non-english (Mandarin) data for training!')

    print('Total clean data:', len(cleanfilenames))
    params['cleanfilenames'] = cleanfilenames
    shuffle(params['cleanfilenames'])
    params['num_cleanfiles'] = len(params['cleanfilenames'])
    # If there are .wav files in noise_dir directory, use those
    # If not, that implies that the noise files are organized into subdirectories by type,
    # so get the names of the non-excluded subdirectories
    if 'noise_csv' in cfg.keys() and cfg['noise_csv'] != 'None':
        noisefilenames = pd.read_csv(cfg['noise_csv'])
        noisefilenames = noisefilenames['filename']
    else:
        noisefilenames = glob.glob(os.path.join(noise_dir, params['audioformat']))

    params['noisefilenames'] = noisefilenames
    shuffle(params['noisefilenames'])

    # rir
    temp = pd.read_csv(params['rir_table_csv'], skiprows=[1], sep=',', header=None,
                       names=['wavfile', 'channel', 'T60_WB', 'C50_WB', 'isRealRIR'])
    temp.keys()
    # temp.wavfile

    rir_wav = temp['wavfile'][1:]  # 115413
    rir_channel = temp['channel'][1:]
    rir_t60 = temp['T60_WB'][1:]
    rir_isreal = temp['isRealRIR'][1:]

    rir_wav2 = [w.replace('\\', '/') for w in rir_wav]
    rir_channel2 = [w for w in rir_channel]
    rir_t60_2 = [w for w in rir_t60]
    rir_isreal2 = [w for w in rir_isreal]

    myrir = []
    mychannel = []
    myt60 = []

    lower_t60 = params['lower_t60']
    upper_t60 = params['upper_t60']

    if params['rir_choice'] == 1:  # real 3076 IRs
        real_indices = [i for i, x in enumerate(rir_isreal2) if x == "1"]

        chosen_i = []
        for i in real_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    elif params['rir_choice'] == 2:  # synthetic 112337 IRs
        synthetic_indices = [i for i, x in enumerate(rir_isreal2) if x == "0"]

        chosen_i = []
        for i in synthetic_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    elif params['rir_choice'] == 3:  # both real and synthetic
        all_indices = [i for i, x in enumerate(rir_isreal2)]

        chosen_i = []
        for i in all_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    else:  # default both real and synthetic
        all_indices = [i for i, x in enumerate(rir_isreal2)]

        chosen_i = []
        for i in all_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    params['myrir'] = myrir
    params['mychannel'] = mychannel
    params['myt60'] = myt60

    # Invoke multiple processes and fan out calls to main_gen() to these processes
    global clean_counter, noise_counter
    clean_counter = multiprocessing.Value('i', 0)
    noise_counter = multiprocessing.Value('i', 0)

    multi_pool = multiprocessing.Pool(processes=PROCESSES, initializer=init, initargs=(clean_counter, noise_counter, ))
    fileindices = range(params['num_files'])
    output_lists = multi_pool.starmap(main_gen, zip(repeat(params), fileindices))

    flat_output_lists = []
    num_lists = 7
    for i in range(num_lists):
        flat_output_lists.append(extract_list(output_lists, i))

    # Create log directory if needed, and write log files of clipped and low activity files
    log_dir = get_dir(cfg, 'log_dir')

    write_log_file(log_dir, 'source_files.csv', flat_output_lists[0] + flat_output_lists[3])
    write_log_file(log_dir, 'clipped_files.csv', flat_output_lists[1] + flat_output_lists[4])
    write_log_file(log_dir, 'low_activity_files.csv', flat_output_lists[2] + flat_output_lists[5])

    flat_output_lists[6].sort(key=lambda x: int(re.findall('\d+', x)[0]))  # 找出字符串中的数字并依据其整形进行排序
    train_list_file = get_file(cfg, 'train_list_file')
    with open(train_list_file, 'w') as lst:
        for i in flat_output_lists[6]:
            lst.write(i + '\n')

    # Compute and print stats about percentange of clipped and low activity files
    total_clean = len(flat_output_lists[0]) + len(flat_output_lists[1]) + len(flat_output_lists[2])
    total_noise = len(flat_output_lists[3]) + len(flat_output_lists[4]) + len(flat_output_lists[5])
    pct_clean_clipped = round(len(flat_output_lists[1]) / total_clean * 100, 1)
    pct_noise_clipped = round(len(flat_output_lists[4]) / total_noise * 100, 1)
    pct_clean_low_activity = round(len(flat_output_lists[2]) / total_clean * 100, 1)
    pct_noise_low_activity = round(len(flat_output_lists[5]) / total_noise * 100, 1)

    print("Of the " + str(total_clean) + " clean speech files analyzed, " + str(pct_clean_clipped) +
          "% had clipping, and " + str(pct_clean_low_activity) + "% had low activity " +
          "(below " + str(params['clean_activity_threshold'] * 100) + "% active percentage)")
    print("Of the " + str(total_noise) + " noise files analyzed, " + str(pct_noise_clipped) +
          "% had clipping, and " + str(pct_noise_low_activity) + "% had low activity " +
          "(below " + str(params['noise_activity_threshold'] * 100) + "% active percentage)")
    



