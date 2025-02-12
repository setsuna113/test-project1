
import os
import torch
import torch.nn as nn
import speechbrain as sb


class STFT_Layer(nn.Module):
    def __init__(self, frame_len, frame_hop):
        super(STFT_Layer, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop

    def forward(self, x):
        if len(x.shape) != 2:
            print("x must be in [B, T]")
        y = torch.stft(x, n_fft=self.frame_len, hop_length=self.frame_hop,
                       win_length=self.frame_len, return_complex=True, center=False)
        r = y.real
        i = y.imag
        # mag=torch.abs(y)
        mag = torch.clamp(r ** 2 + i ** 2, self.eps) ** 0.5
        phase = torch.atan2(i + self.eps, r + self.eps)
        return mag, phase


class InstantLayerNormalization(nn.Module):
    """
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    """

    def __init__(self, channels):
        """
            Constructor
        """
        super(InstantLayerNormalization, self).__init__()
        self.epsilon = 1e-7
        self.gamma = nn.Parameter(torch.ones(1, 1, channels), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, 1, channels), requires_grad=True)
        self.register_parameter("gamma", self.gamma)
        self.register_parameter("beta", self.beta)

    def forward(self, inputs):
        # calculate mean of each frame
        mean = torch.mean(inputs, dim=-1, keepdim=True)

        # calculate variance of each frame
        variance = torch.mean(torch.square(inputs - mean), dim=-1, keepdim=True)
        # calculate standard deviation
        std = torch.sqrt(variance + self.epsilon)
        # normalize each frame independently
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs


class SeperationBlock(nn.Module):
    def __init__(self, input_size=257, hidden_size=128, dropout=0.25):
        super(SeperationBlock, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.rnn2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.drop = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, (h, c) = self.rnn1(x)
        x1 = self.drop(x1)
        x2, _ = self.rnn2(x1)
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)
        return mask


class SeperationBlock_Stateful(nn.Module):
    def __init__(self, input_size=257, hidden_size=128, dropout=0.25):
        super(SeperationBlock_Stateful, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.rnn2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.drop = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, in_states):
        """

        :param x:  [N, T, input_size]
        :param in_states: [2, N, hidden_size, 2]
        :return:
        """
        h1_in, c1_in = in_states[:1, :, :, 0], in_states[:1, :, :, 1]
        h2_in, c2_in = in_states[1:, :, :, 0], in_states[1:, :, :, 1]

        x1, (h1, c1) = self.rnn1(x, (h1_in, c1_in))
        x1 = self.drop(x1)
        x2, (h2, c2) = self.rnn2(x1, (h2_in, c2_in))
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)

        h = torch.cat((h1, h2), dim=0)
        c = torch.cat((c1, c2), dim=0)
        out_states = torch.stack((h, c), dim=-1)
        return mask, out_states


class DTLN(nn.Module):
    def __init__(self, frame_len=512, frame_hop=128, window='rect'):
        super(DTLN, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.stft = STFT_Layer(frame_len, frame_hop)

        self.sep1 = SeperationBlock(input_size=(frame_len // 2 + 1), hidden_size=128, dropout=0.25)

        self.encoder_size = 256
        self.encoder_conv1 = nn.Conv1d(in_channels=frame_len, out_channels=self.encoder_size,
                                       kernel_size=1, stride=1, bias=False)

        # self.encoder_norm1 = nn.InstanceNorm1d(num_features=self.encoder_size, eps=1e-7, affine=True)
        self.encoder_norm1 = InstantLayerNormalization(channels=self.encoder_size)

        self.sep2 = SeperationBlock(input_size=self.encoder_size, hidden_size=128, dropout=0.25)

        # TODO with causal padding like in keras,when ksize > 1
        self.decoder_conv1 = nn.Conv1d(in_channels=self.encoder_size, out_channels=frame_len,
                                       kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        """

        :param x:  [N, T]
        :return:
        """
        batch, n_frames = x.shape

        mag, phase = self.stft(x)
        mag = mag.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)

        # N, T, hidden_size
        mask = self.sep1(mag)
        estimated_mag = mask * mag

        s1_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft2(s1_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)

        mask_2 = self.sep2(encoded_f_norm)
        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0, 2, 1)

        decoded_frame = self.decoder_conv1(estimated)

        # overlap and add
        out = torch.nn.functional.fold(
            decoded_frame,
            (n_frames, 1),
            kernel_size=(self.frame_len, 1),
            padding=(0, 0),
            stride=(self.frame_hop, 1),
        )
        out = out.reshape(batch, -1)

        return out

    def forward(self, x):
        """

        :param x:  [N, T]
        :return:
        """
        batch, n_frames = x.shape

        mag, phase = self.stft(x)
        mag = mag.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)

        # N, T, hidden_size
        mask = self.sep1(mag)
        estimated_mag = mask * mag

        s1_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft2(s1_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)

        mask_2 = self.sep2(encoded_f_norm)
        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0, 2, 1)

        decoded_frame = self.decoder_conv1(estimated)

        # overlap and add
        out = torch.nn.functional.fold(
            decoded_frame,
            (n_frames, 1),
            kernel_size=(self.frame_len, 1),
            padding=(0, 0),
            stride=(self.frame_hop, 1),
        )
        out = out.reshape(batch, -1)

        return out

class DTLN_stateful(nn.Module):
    def __init__(self, frame_len=512, frame_hop=128, window='rect'):
        super(DTLN_stateful, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.stft = STFT_Layer(frame_len, frame_hop)

        self.sep1 = SeperationBlock_Stateful(input_size=(frame_len // 2 + 1), hidden_size=128, dropout=0.25)

        self.encoder_size = 256
        self.encoder_conv1 = nn.Conv1d(in_channels=frame_len, out_channels=self.encoder_size,
                                       kernel_size=1, stride=1, bias=False)

        # self.encoder_norm1 = nn.InstanceNorm1d(num_features=self.encoder_size, eps=1e-7, affine=True)
        self.encoder_norm1 = InstantLayerNormalization(channels=self.encoder_size)

        self.sep2 = SeperationBlock_Stateful(input_size=self.encoder_size, hidden_size=128, dropout=0.25)

        # TODO with causal padding like in keras,when ksize > 1
        self.decoder_conv1 = nn.Conv1d(in_channels=self.encoder_size, out_channels=frame_len,
                                       kernel_size=1, stride=1, bias=False)

    def forward(self, x, in_state1, in_state2):
        """

        :param x:  [N, T]
        :return:
        """
        batch, n_frames = x.shape
        assert n_frames == self.frame_len

        mag, phase = self.stft(x)
        mag = mag.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)

        # N, T, hidden_size
        mask, out_state1 = self.sep1(mag, in_state1)
        estimated_mag = mask * mag

        s1_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft2(s1_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask_2, out_state2 = self.sep2(encoded_f_norm, in_state2)
        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0, 2, 1)
        decoded_frame = self.decoder_conv1(estimated)
        return decoded_frame, out_state1, out_state2


class DTLN_P1_stateful(nn.Module):
    def __init__(self, frame_len=512, frame_hop=128, window='rect'):
        super(DTLN_P1_stateful, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop

        self.sep1 = SeperationBlock_Stateful(input_size=(frame_len // 2 + 1), hidden_size=128, dropout=0.25)

    def forward(self, mag, in_state1):
        """

        :param mag:  [b, T, 257]
        :param in_state1: [2, b, 128, 2]
        :return:
        """
        # N, T, hidden_size
        mask, out_state1 = self.sep1(mag, in_state1)
        estimated_mag = mask * mag
        return estimated_mag, out_state1


class DTLN_P2_stateful(nn.Module):
    def __init__(self, frame_len=512):
        super(DTLN_P2_stateful, self).__init__()
        self.frame_len = frame_len
        self.encoder_size = 256
        self.encoder_conv1 = nn.Conv1d(in_channels=frame_len, out_channels=self.encoder_size,
                                       kernel_size=1, stride=1, bias=False)

        # self.encoder_norm1 = nn.InstanceNorm1d(num_features=self.encoder_size, eps=1e-7, affine=True)
        self.encoder_norm1 = InstantLayerNormalization(channels=self.encoder_size)

        self.sep2 = SeperationBlock_Stateful(input_size=self.encoder_size, hidden_size=128, dropout=0.25)

        # TODO with causal padding like in keras,when ksize > 1
        self.decoder_conv1 = nn.Conv1d(in_channels=self.encoder_size, out_channels=frame_len,
                                       kernel_size=1, stride=1, bias=False)

    def forward(self, y1, in_state2):
        """
        :param y1: [b, framelen, T]
        :param in_state2:  [2, b, 128, 2]
        :return:
        """

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)

        mask_2, out_state2 = self.sep2(encoded_f_norm, in_state2)
        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0, 2, 1)

        decoded_frame = self.decoder_conv1(estimated)

        return decoded_frame, out_state2


def test_stateful():
    bsize = 1
    x = torch.randn(bsize, 512)
    # 2, bsize, hidden_size, 2
    in_state1 = torch.randn(2, bsize, 128, 2)
    in_state2 = torch.randn(2, bsize, 128, 2)

    net = DTLN_stateful()
    import tqdm
    for i in tqdm.tqdm(range(100)):
        y, out_state1, out_state2 = net(x, in_state1, in_state2)

    print(y.shape)
    print(out_state1.shape)
    print(out_state2.shape)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        help="model dir",
                        default=os.path.dirname(__file__) + "/results/4234/save/CKPT+2024-01-10+13-48-08+00/model.ckpt")

    args = parser.parse_args()
    state_dict = torch.load(args.model_path)
    for k in state_dict.keys():
        print(k, state_dict[k].shape)

# sep1.rnn1.weight_ih_l0 torch.Size([512, 257])
# sep1.rnn1.weight_hh_l0 torch.Size([512, 128])
# sep1.rnn1.bias_ih_l0 torch.Size([512])
# sep1.rnn1.bias_hh_l0 torch.Size([512])
# sep1.rnn2.weight_ih_l0 torch.Size([512, 128])
# sep1.rnn2.weight_hh_l0 torch.Size([512, 128])
# sep1.rnn2.bias_ih_l0 torch.Size([512])
# sep1.rnn2.bias_hh_l0 torch.Size([512])
# sep1.dense.weight torch.Size([257, 128])
# sep1.dense.bias torch.Size([257])
# encoder_conv1.weight torch.Size([256, 512, 1])
# encoder_norm1.gamma torch.Size([1, 1, 256])
# encoder_norm1.beta torch.Size([1, 1, 256])
# sep2.rnn1.weight_ih_l0 torch.Size([512, 256])
# sep2.rnn1.weight_hh_l0 torch.Size([512, 128])
# sep2.rnn1.bias_ih_l0 torch.Size([512])
# sep2.rnn1.bias_hh_l0 torch.Size([512])
# sep2.rnn2.weight_ih_l0 torch.Size([512, 128])
# sep2.rnn2.weight_hh_l0 torch.Size([512, 128])
# sep2.rnn2.bias_ih_l0 torch.Size([512])
# sep2.rnn2.bias_hh_l0 torch.Size([512])
# sep2.dense.weight torch.Size([256, 128])
# sep2.dense.bias torch.Size([256])
# decoder_conv1.weight torch.Size([512, 256, 1])
