import os
from datasets import Dataset, concatenate_datasets
import argparse
import numpy as np
from demucs.apply import apply_model
from demucs.audio import convert_audio
from datasets import Audio
import torch


# DATASET FEATURES/COLUMNS = ['TRACK_ID', 'ARTIST_ID', 'ALBUM_ID', 'AUDIO', 'DURATION', 'TAGS', 'INSTR', 'GENRE']

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_proc", type=int, default=1)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--audio_separation", type=bool, default=False)


data_args = parser.parse_args()



try:
    from demucs import pretrained
except ImportError:
    print(
        "To perform audio separation, you should install additional packages, run: `pip install -e .[metadata]]` or `pip install demucs`."
    )

demucs = pretrained.get_model("htdemucs")
if torch.cuda.device_count() > 0:
    demucs.to("cuda:0")


def wrap_audio(audio, sr):
    return {"array": audio.cpu().numpy(), "sampling_rate": sr}

def remove_vocals(batch, rank=None):
    device = "cpu" if torch.cuda.device_count() == 0 else "cuda:0"
    if rank is not None:
        # move the model to the right GPU if not there already
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        # move to device and create pipeline here because the pipeline moves to the first GPU it finds anyway
        demucs.to(device)

    if isinstance(batch[audio_column_name], list):
        wavs = [
            convert_audio(
                torch.tensor(audio["array"][None], device=device).to(
                    torch.float32
                ),
                audio["sampling_rate"],
                demucs.samplerate,
                demucs.audio_channels,
            ).T
            for audio in batch["AUDIO"]
        ]
        wavs_length = [audio.shape[0] for audio in wavs]

        wavs = torch.nn.utils.rnn.pad_sequence(
            wavs, batch_first=True, padding_value=0.0
        ).transpose(1, 2)
        stems = apply_model(demucs, wavs)

        batch[audio_column_name] = [
            wrap_audio(s[:-1, :, :length].sum(0).mean(0), demucs.samplerate)
            for (s, length) in zip(stems, wavs_length)
        ]

    else:
        audio = torch.tensor(
            batch[audio_column_name]["array"].squeeze(), device=device
        ).to(torch.float32)
        sample_rate = batch[audio_column_name]["sampling_rate"]
        audio = convert_audio(
            audio, sample_rate, demucs.samplerate, demucs.audio_channels
        )
        stems = apply_model(demucs, audio[None])

        batch[audio_column_name] = wrap_audio(
            stems[0, :-1].mean(0), demucs.samplerate
        )

    return batch


base_dir = "/share/cp/temp/musicmed/metadata_test_32k/"
arrow_files = [os.path.join(base_dir, fn) for fn in os.listdir(base_dir) if fn.endswith(".arrow") and fn.startswith("data")]
ds = concatenate_datasets([Dataset.from_file(arrow_file) for arrow_file in arrow_files])

audio_column_name = "AUDIO"

# set sharing strategy
torch.multiprocessing.set_sharing_strategy("file_system")
# set multiprocessing start method
torch.multiprocessing.set_start_method("spawn")


# find if track has voice
if "has_voice" not in ds.column_names:
    has_voice = np.char.find(ds["INSTR"], "voice") > -1
    ds = ds.add_column("has_voice", has_voice)
else:
    ds = ds.remove_columns("has_voice")
    has_voice = ds["has_voice"]

# find idx of tracks with voice
voice_idx = np.where(has_voice)[0]

# filter, merge and create metadata
# for every entry add space after comma
instr = ds["INSTR"]
instr = np.char.replace(instr, ",", ", ")

tags = ds["TAGS"]
tags = np.char.replace(tags, ",", ", ")

genre = ds["GENRE"]
genre = np.char.replace(genre, ",", ", ")

# merge instr, tags and genre into metadata with comma separated values
metadata = np.char.add(np.char.add(instr, tags), genre)
ds = ds.add_column("METADATA", metadata)

# find all unique tag words
tags = np.char.split(tags, ", ")
unique_tags = np.unique(np.concatenate(tags))
# remove empty strings
unique_tags = unique_tags[unique_tags != ""]

ds_voice = ds.select(voice_idx)


# filter out tracks with voice
ds_voice = ds_voice.map(
            remove_vocals,
            batched=True,
            batch_size=data_args.batch_size,
            with_rank=True,
            num_proc=data_args.num_proc,
        )

# select dataset indices without voice
ds_no_voice = ds.select(np.where(~has_voice)[0])

# merge datasets
ds = concatenate_datasets([ds_voice, ds_no_voice])

# save dataset
ds.save_to_disk("/share/cp/temp/musicmed/metadata_test_32k_no_voice", num_proc=20, max_shard_size="2GB")

