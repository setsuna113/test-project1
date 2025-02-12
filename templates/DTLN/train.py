#!/usr/bin/env/python3
"""Recipe for training a speech enhancement system with spectral masking.

To run this recipe, do the following:
> python train.py train.yaml --data_folder /path/to/save/mini_librispeech

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

Authors
 * Szu-Wei Fu 2020
 * Chien-Feng Liao 2020
 * Peter Plantinga 2021
"""
import os, shutil
import torch
import shutil
import librosa
import speechbrain as sb
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
import time

from dns.utils import datasets_untar_path
# from dns.agc_data_prepare import agc_synthetic
from dns.metrics import metrics_visualization
from dns.audiolib import audiowrite
from dataloader import train_dataloader, valid_dataloader, test_dataloader
from data_prepare import prepare_dns_noisy, check_folders
# Brain class for speech enhancement training


class DTLN(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Apply masking to convert from noisy waveforms to enhanced signals.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            A dictionary with keys {"spec", "wav"} with predicted features.
        """

        # We first move the batch to the appropriate device, and
        # compute the features necessary for masking.

        batch['noisy'] = batch['noisy'].to(self.device)
        batch['clean'] = batch['clean'].to(self.device)

        # 送入模型
        result_list = self.modules.model(batch['noisy'])

        output = {
            'estimated_sig': result_list,
        }

        # Return a dictionary so we don't have to remember the order
        return output

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        # 计算信噪比作为loss
        estimated_sig = predictions['estimated_sig'].unsqueeze(0).unsqueeze(0).permute(3, 0, 2, 1)
        clean = batch['clean'].unsqueeze(0).unsqueeze(0).permute(3, 0, 2, 1)
        loss = sb.nnet.losses.cal_snr(clean, estimated_sig)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch['idxs'], predictions["estimated_sig"], batch['clean'], reduction="batch"
        )
        if stage == sb.Stage.TEST:
            self.metric_data(predictions['estimated_sig'], batch)

        return torch.mean(loss)

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.mse_loss
        )

        if stage == sb.Stage.TEST:
            self.dump_data = {
                'With_reverb': {
                    'clean': [],
                    'noisy': [],
                    'enhanced': [],
                },
                'No_reverb': {
                    'clean': [],
                    'noisy': [],
                    'enhanced': [],
                }
            }

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
            }

        # At the end of validation, we can write stats and checkpoints
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best STOI score.
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            checkpoint_name = f"checkpoint_epoch{epoch}_loss{stage_loss:.4f}_{timestamp}"
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            # 收集完测试数据开始计算语音质量评价指标
            for speech_type in ("With_reverb", "No_reverb"):
                if len(self.dump_data[speech_type]['noisy']) == 0 or \
                        len(self.dump_data[speech_type]['clean']) == 0 or \
                        len(self.dump_data[speech_type]['enhanced']) == 0:
                    continue
                print(f'Performance Checking <{speech_type}>...')
                mean_score_dict = metrics_visualization(
                    self.dump_data[speech_type]['noisy'],
                    self.dump_data[speech_type]['clean'],
                    self.dump_data[speech_type]['enhanced'],
                    self.hparams.metrics_list,
                    epoch,
                    self.hparams.num_workers,
                    mark=speech_type,
                )

                stoi_mean = mean_score_dict['STOI']
                # Transform PESQ metric range from [-0.5 ~ 4.5] to [0 ~ 1]
                wb_pesq_mean = (mean_score_dict['WB_PESQ'] + 0.5) / 5
                avg_score = (stoi_mean + wb_pesq_mean) / 2

                print(f'{speech_type}: {mean_score_dict}')
                print(f'avg_score: {avg_score}')

    def metric_data(self, enhanced_s, batch):
        for id in range(len(batch['idxs'])):
            if batch['mode'][id] == 'test_metric':
                clean = batch['clean'][id].detach().squeeze(0).cpu().numpy()
                noisy = batch['noisy'][id].detach().squeeze(0).cpu().numpy()
                enhanced = enhanced_s[id].detach().squeeze(0).cpu().numpy()

                # pesq测试指标只支持16k
                if self.hparams.sample_rate != 16000:
                    noisy = librosa.resample(noisy, orig_sr=self.hparams.sample_rate, target_sr=16000)
                    clean = librosa.resample(clean, orig_sr=self.hparams.sample_rate, target_sr=16000)
                    enhanced = librosa.resample(enhanced, orig_sr=self.hparams.sample_rate, target_sr=16000)

                speech_type = batch['speech_type'][id]
                self.dump_data[speech_type]['clean'].append(clean)
                self.dump_data[speech_type]['noisy'].append(noisy)
                self.dump_data[speech_type]['enhanced'].append(enhanced)

                # 保存路径
                clean_src_path = batch['clean_path'][id]
                noisy_src_path = batch['noisy_path'][id]
                save_dir = os.path.join(self.hparams.data_folder, 'test_metric_out', speech_type)
                filename, ext = os.path.splitext(os.path.basename(noisy_src_path))

                clean_dst_path = os.path.join(save_dir, filename + '_clean' + ext)                  # clean
                noisy_dst_path = os.path.join(save_dir, filename + '_noisy' + ext)                  # noisy
                enhanced_dst_path = os.path.join(save_dir, filename + '_enhanced' + ext)            # enhanced

                Path(os.path.dirname(clean_dst_path)).mkdir(parents=True, exist_ok=True)
                Path(os.path.dirname(noisy_dst_path)).mkdir(parents=True, exist_ok=True)
                Path(os.path.dirname(enhanced_dst_path)).mkdir(parents=True, exist_ok=True)

                shutil.copy(clean_src_path, clean_dst_path)
                shutil.copy(noisy_src_path, noisy_dst_path)
                audiowrite(enhanced_dst_path, enhanced, 16000)
            if batch['mode'][id] == 'test':
                clean = batch['clean'][id].detach().squeeze(0).cpu().numpy()
                noisy = batch['noisy'][id].detach().squeeze(0).cpu().numpy()
                enhanced = enhanced_s[id].detach().squeeze(0).cpu().numpy()

                # 保存路径
                clean_src_path = batch['clean_path'][id]
                noisy_src_path = batch['noisy_path'][id]
                save_dir = os.path.join(self.hparams.data_folder, 'test_out')
                filename, ext = os.path.splitext(os.path.basename(noisy_src_path))

                clean_dst_path = os.path.join(save_dir, filename + '_clean' + ext)                  # clean
                noisy_dst_path = os.path.join(save_dir, filename + '_noisy' + ext)                  # noisy
                enhanced_dst_path = os.path.join(save_dir, filename + '_enhanced' + ext)            # enhanced

                Path(os.path.dirname(clean_dst_path)).mkdir(parents=True, exist_ok=True)
                Path(os.path.dirname(noisy_dst_path)).mkdir(parents=True, exist_ok=True)
                Path(os.path.dirname(enhanced_dst_path)).mkdir(parents=True, exist_ok=True)

                shutil.copy(clean_src_path, clean_dst_path)
                shutil.copy(noisy_src_path, noisy_dst_path)
                audiowrite(enhanced_dst_path, enhanced, self.hparams.sample_rate)


# Recipe begins!
if __name__ == "__main__":
    curr_path = os.path.abspath(os.path.dirname(__file__))
    project_name = Path(curr_path).name
    arg_list = [
        f'{curr_path}/train.yaml',
        '--curr_path', f'{curr_path}',
        '--project_name', f'{project_name}',
        '--device', 'cuda:0',
        '--data_untar_folder', datasets_untar_path(),
    ]

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(arg_list)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    #if os.path.exists(hparams['data_folder']):
        #shutil.rmtree(hparams['data_folder'])
    if not check_folders(hparams['data_folder']):
        sb.utils.distributed.run_on_main(
            prepare_dns_noisy,
            kwargs={
                "noisy_speech": hparams["noisy_speech"],
                "save_csv_train": hparams["train_annotation"],
                "save_csv_valid": hparams["valid_annotation"],
                "save_csv_test": hparams["test_annotation"],
            },
        )

    # Initialize the Brain object to prepare for mask training.
    se_brain = DTLN(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    if not run_opts["test_only"]:
        se_brain.fit(
            epoch_counter=se_brain.hparams.epoch_counter,
            train_set=train_dataloader(hparams),
            valid_set=valid_dataloader(hparams),
        )

    # Load best checkpoint (highest STOI) for evaluation
    test_stats = se_brain.evaluate(
        test_set=test_dataloader(hparams),
        min_key="loss",
    )
