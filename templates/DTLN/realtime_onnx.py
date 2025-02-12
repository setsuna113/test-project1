

import os
import argparse
import numpy as np
import onnxruntime
from hyperpyyaml import load_hyperpyyaml
from speechbrain.expand.DNS_Challenge.audiolib import realtime_device_task, realtime_wav_task


class ModelInfer:
    def __init__(self, model1_path, model2_path,
                 block_len=512, block_shift=128) -> None:

        print('==> load model1 from: ', model1_path)
        # load models
        self.interpreter_1 = onnxruntime.InferenceSession(model1_path)
        self.model_input_names_1 = [inp.name for inp in self.interpreter_1.get_inputs()]
        # preallocate input
        self.model_inputs_1 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in self.interpreter_1.get_inputs()}

        for item in self.model_inputs_1.items():
            print("[ model1 ] input {} , shape: {}".format(item[0], item[1].shape))

        print('==> load model2 from: ', model2_path)
        self.interpreter_2 = onnxruntime.InferenceSession(model2_path)
        self.model_input_names_2 = [inp.name for inp in self.interpreter_2.get_inputs()]
        # preallocate input
        self.model_inputs_2 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in self.interpreter_2.get_inputs()}

        for item in self.model_inputs_2.items():
            print("[ model2] {} , shape: {}".format(item[0], item[1].shape))

        # create buffer
        self.in_buffer = np.zeros((block_len))
        self.out_buffer = np.zeros((block_len))

        self.block_shift = block_shift

    def process(self, data_in):
        # shift values and write to buffer
        self.in_buffer[:-self.block_shift] = self.in_buffer[self.block_shift:]
        self.in_buffer[-self.block_shift:] = data_in[:]
        # in_block = np.expand_dims(self.in_buffer, axis=0).astype('float32')

        in_block_fft = np.fft.rfft(self.in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        # reshape magnitude to input dimensions
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')

        # set block to input
        self.model_inputs_1[self.model_input_names_1[0]] = in_mag
        # run calculation
        model_outputs_1 = self.interpreter_1.run(None, self.model_inputs_1)
        # get the output of the first block
        estimated_mag = model_outputs_1[0]

        # set out states back to input
        self.model_inputs_1["h1_in"][0] = model_outputs_1[1][0]
        self.model_inputs_1["c1_in"][0] = model_outputs_1[1][1]
        self.model_inputs_1["h2_in"][0] = model_outputs_1[1][2]
        self.model_inputs_1["c2_in"][0] = model_outputs_1[1][3]

        # calculate the ifft
        estimated_complex = estimated_mag * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        # reshape the time domain block
        estimated_block = np.reshape(estimated_block, (1, -1, 1)).astype('float32')
        # set tensors to the second block
        # interpreter_2.set_tensor(input_details_1[1]['index'], states_2)
        self.model_inputs_2[self.model_input_names_2[0]] = estimated_block
        # run calculation
        model_outputs_2 = self.interpreter_2.run(None, self.model_inputs_2)
        # get output
        out_block = model_outputs_2[0]
        # set out states back to input

        self.model_inputs_2["h1_in"][0] = model_outputs_2[1][0]
        self.model_inputs_2["c1_in"][0] = model_outputs_2[1][1]
        self.model_inputs_2["h2_in"][0] = model_outputs_2[1][2]
        self.model_inputs_2["c2_in"][0] = model_outputs_2[1][3]

        # shift values and write to buffer
        self.out_buffer[:-self.block_shift] = self.out_buffer[self.block_shift:]
        self.out_buffer[-self.block_shift:] = np.zeros((self.block_shift))
        self.out_buffer += np.squeeze(out_block)

        # print(idx, np.abs(out_buffer).sum())
        # write block to output file
        return self.out_buffer[:self.block_shift]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--process_mode',
                        type=str,
                        help='processing mode (wav/device)',
                        default='device')

    parser.add_argument('--hp_config',
                        type=str,
                        help='hyperparameters config',
                        default=os.path.join(os.path.dirname(__file__), "results/4234/hyperparams.yaml"))

    parser.add_argument("--model1_path",
                        type=str,
                        help="model1 path",
                        default=os.path.join(os.path.dirname(__file__), "pretrained/model_p1.onnx"))

    parser.add_argument("--model2_path",
                        type=str,
                        help="model2 path",
                        default=os.path.join(os.path.dirname(__file__), "pretrained/model_p2.onnx"))

    parser.add_argument("--wav_in",
                        type=str,
                        help="wav in",
                        default=os.path.join(os.path.dirname(__file__), "samples/audioset_realrec_airconditioner_2TE3LoA2OUQ.wav"))

    parser.add_argument("--wav_out",
                        type=str,
                        help="wav out",
                        default=os.path.join(os.path.dirname(__file__), "samples/enhanced_onnx.wav"))
    args = parser.parse_args()

    # Load hyperparameters file with command-line overrides
    with open(args.hp_config) as fin:
        hparams = load_hyperpyyaml(fin)
    # �������м��ز���
    sample_rate = hparams['sample_rate']
    block_len = hparams['win_length']
    block_shift = hparams['hop_length']

    model_infer = ModelInfer(args.model1_path, args.model2_path, block_len, block_shift)

    if args.process_mode == 'device':
        realtime_device_task(model_infer, sample_rate, block_len, block_shift)
    elif args.process_mode == 'wav':
        realtime_wav_task(model_infer, args.wav_in, args.wav_out, sample_rate, block_len, block_shift)
