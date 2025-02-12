import os
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from model import DTLN
import torch
import pandas as pd
from dataloader import valid_dataloader
import torchaudio
from datetime import datetime
import librosa
import shutil


# Initialize the device and yaml
if torch.cuda.is_available():
    try:
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Failed to initialize CUDA device: {e}")
        device = torch.device("cpu")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

curr_path = os.path.abspath(os.path.dirname(__file__))
hparams_file = os.path.join(curr_path, "train.yaml")
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin)

# load model from ckpt
model = DTLN(
    frame_len=hparams["win_length"],
    frame_hop=hparams["hop_length"],
    window=hparams["window_fn"],
).to(device)
'''
print("Model state before manual recovery:", list(model.parameters())[0].data)
checkpointer = hparams["checkpointer"]
#checkpointer.checkpoints_dir = Path(os.path.join(curr_path, checkpointer.checkpoints_dir))
result = checkpointer.recover_if_possible()
print("Model state after manual recovery:", list(model.parameters())[0].data)
if result is None:
    raise ValueError("Recovery failed: checkpointer.recover_if_possible() returned None")
model.eval()
'''

ckpt_path = hparams["checkpointer"].checkpoints_dir
#ckpt_path = os.path.join(curr_path, ckpt_path)
if not os.path.exists(ckpt_path):
    print(f"Checkpoint path {ckpt_path} does not exist. Exiting...")
    exit(1)

ckpt_list = []
# Walk through the root directory and its subdirectories
for root, dirs, files in os.walk(ckpt_path):
    for file in files:
        if file.endswith("model.ckpt"):
            full_path = os.path.join(root, file)
            ckpt_list.append(full_path)
ckpt_dir = ckpt_list[0]
ckpt = torch.load(ckpt_dir, weights_only=True)
model.load_state_dict(ckpt)

# csv and output path
csv_path = os.path.join(hparams["data_folder"], "valid.csv")
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(hparams["output_folder"], f"processed_{timestamp}")
os.makedirs(out_path, exist_ok=True)
df = pd.read_csv(csv_path)

# process the audio
for i, row in df.iterrows():
    audio_path = row["noisy_path"]
    audio_output = os.path.join(out_path, f"processed_{i}.wav")
    
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.to(device)

    with torch.no_grad():
        processed = model(waveform)
        if processed.ndim != 2:
            raise ValueError(f"Processed output is not a 2D tensor: {processed.shape}")

    processed = processed.cpu()
    torchaudio.save(
        uri = audio_output, 
        src = processed, 
        sample_rate = sr,
        #format = "wav",
    )