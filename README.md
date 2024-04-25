# MusicGen Dreamboothing

This repository contains lightweight training code for [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md), a state-of-the-art controllable text-to-music model. MusicGen is a single stage auto-regressive Transformer model trained over a [32kHz Encodec tokenizer](https://huggingface.co/facebook/encodec_32khz) with 4 codebooks sampled at 50 Hz.

The aim is to provide **tools** to **easily fine-tune** and **dreambooth** the MusicGen model suite on **small consumer GPUs**, with little data and to leverage a series of optimizations and tricks to reduce resource consumption. For example, the model can be fine-tuned on a particular music genre or artist to give a checkpoint that generates in that given style. The aim is also to easily **share and build** on these trained checkpoints, thanks to [LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora#lora) adaptors.

Specifically, this involves:
* using as few data and resources as possible. We're talking fine-tuning with as little as 15mn on an A100 and as little as 10GB to 16GB of GPU utilization.
* easily share and build models thanks to the [Hugging Face Hub](https://huggingface.co/models).
* optionally, generate automatic music descriptions
* optionally, training MusicGen in a [Dreambooth](https://huggingface.co/docs/diffusers/en/training/dreambooth)-like fashion, where one key-word triggers generation in a particular style


## 📖 Quick Index
* [Requirements](#requirements)
* [Training](#training)
* [Inference](#inference)
* [❓ FAQ](#faq)

## Requirements

You first need to clone this repository before installing requirements.

```sh
git clone git@github.com:ylacombe/musicgen-dreamboothing.git
cd musicgen-dreamboothing
pip install -e .
pip install -U git+https://github.com/huggingface/transformers
```
> [!NOTE]
> For the time being, we're installing `transformers` from source, but you won't have to do once the next version is released anytime soon. 

Optionally, you can create a wandb account and login to it by following [this guide](https://docs.wandb.ai/quickstart). [`wandb`](https://docs.wandb.ai/) allows for better tracking of the experiments metrics and losses.

You also have the option to configure Accelerate by running the following command. Note that you should set the number of GPUs you wish to use for training, and also the data type (dtype) to your preferred dtype for training/inference (e.g. `bfloat16` on A100 GPUs, `float16` on V100 GPUs, etc.):

```bash
accelerate config
```

Lastly, you can link you Hugging Face account so that you can push model repositories on the Hub. This will allow you to save your trained models on the Hub so that you can share them with the community. Run the command:

```bash
git config --global credential.helper store
huggingface-cli login
```

And then enter an authentication token from https://huggingface.co/settings/tokens. Create a new token if you do not have one already. You should make sure that this token has "write" privileges.

## Training

### Training guide

The script [`dreambooth_musicgen.py`](dreambooth_musicgen.py) is an end-to-end script that:
1. Loads an audio dataset using the [`datasets`](https://huggingface.co/docs/datasets/v2.17.0/en/index) library, for example this [small subset of songs in the punk style](https://huggingface.co/datasets/ylacombe/tiny-punk) derived from the royalty-free [PogChamp Music Classification Competition](https://www.kaggle.com/competitions/kaggle-pog-series-s01e02/overview) dataset.
2. Loads a MusicGen checkpoint from the hub, for example the [1.5B MusicGen Melody checkpoint](https://huggingface.co/facebook/musicgen-melody).
3. (Optional) Generates automatic song descriptions with the `--add_metadata true` and `--instance_prompt ANCHOR_STRING` flags. This will automatically computes instruments, genre, mood, tempo and key, as well as add the instance prompt to the description. In the following example, the instance prompt is `punk` because we use a dataset made of punk songs. 
4. (Optional) Get rid of lyrics with the flag `--audio_separation`. MusicGen was trained on instrumentals only, so lyrics will only degrade the model training.
5. Tokenizes the text descriptions and encode the audio samples.
6. Uses the [Transformers' Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) to perform training and evaluation.

> [!IMPORTANT]
> If you want to generate automatic song descriptions or if you want to do audio separation, you should also install optional dependencies with the following: `pip install -e .[metadata]`  

You can learn more about the different arguments of the training script by running:

```sh
python dreambooth_musicgen.py --help
```

To give a practical example, here's how to fine-tune [MusicGen Melody](https://huggingface.co/facebook/musicgen-melody) on 27 minutes of [Punk music](https://huggingface.co/datasets/ylacombe/tiny-punk/viewer/default/clean).

```sh
python dreambooth_musicgen.py \
    --use_lora \
    --model_name_or_path "facebook/musicgen-melody" \
    --dataset_name "ylacombe/tiny-punk" \
    --dataset_config_name "default" \
    --target_audio_column_name "others" \
    --instance_prompt "punk" \
    --train_split_name "clean" \
    --eval_split_name "clean" \
    --output_dir "./musicgen-melody-lora-punk" \
    --do_train \
    --fp16 \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.1 \
    --guidance_scale 3.0 \
    --do_eval \
    --predict_with_generate \
    --include_inputs_for_metrics \
    --eval_steps 25 \
    --evaluation_strategy "steps" \
    --per_device_eval_batch_size 1 \
    --max_eval_samples 8 \
    --generation_max_length 400 \
    --dataloader_num_workers 8 \
    --logging_steps 1 \
    --max_duration_in_seconds 30 \
    --min_duration_in_seconds 1.0 \
    --preprocessing_num_workers 8 \
    --pad_token_id 2048 \
    --decoder_start_token_id 2048 \
    --seed 456 \
    --overwrite_output_dir \
    --push_to_hub true
```

Using a few tricks, this fine-tuning run used 10GB of GPU memory and ran in under 15 minutes on an A100 GPU. The resulting
punk checkpoint can be found on the Hugging Face Hub under [ylacombe/musicgen-melody-lora-punk](https://huggingface.co/ylacombe/musicgen-melody-lora-punk).

More specifically, those tricks are [LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora#lora), [half-precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format), [gradient accumulation](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps) and [gradient checkpointing](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.gradient_checkpointing). The largest memory saving comes from LoRA, which is a training technique for significantly reducing the number of trainable parameters. As a result, training is faster and it is easier to store the resulting weights because they are a lot smaller (~100MBs). For more information, refer to the [LoRA conceptual guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora). Note that using the same parameters on [MusicGen Melody large](https://huggingface.co/facebook/musicgen-melody-large) only used 16GB of GPU.

Also note that you can also use a JSON file to get your parameters. For example, [punk.json](/example_configs/punk.json):

```sh
python dreambooth_musicgen.py example_configs/punk.json
```

The JSON example above also shows to follow the training thanks to wandb (e.g of what it looks like [here](https://wandb.ai/ylacombe/musicgen_finetuning_experiments/runs/lk6x8k4u?nw=nwuserylacombe)).

### Tips

Some take-aways from the different experiments we've done:
* to fine-tune and keep model ability it's essential to have a low number of epochs.
* for small datasets, a learning rate of 2e-4 gave great results.
* it doesn't actually matter to have the training loss going down, it's always better to actually listen to the output samples.
* you can get quickly get a sense of how and if the model learned by comparing the samples before and after fine-tuning on wandb (e.g [here](https://wandb.ai/ylacombe/musicgen_finetuning_experiments/runs/lk6x8k4u?nw=nwuserylacombe)).
* If you're not using a melody checkpoint and get `nan` errors, you might want to set `guidance_scale` to 1.0, check this [FAQ response](#im-getting-nan-errors-with-some-checkpoints-what-do-i-do).

## Inference

The following snippet indicates how to do inference with LoRA fine-tuned MusicGen model:

```python
from peft import PeftConfig, PeftModel
from transformers import AutoModelForTextToWaveform, AutoProcessor
import torch
import soundfile as sf

device = torch.device("cuda:0" if torch.cuda.device_count()>0 else "cpu")

repo_id = "ylacombe/musicgen-melody-punk-lora"

config = PeftConfig.from_pretrained(repo_id)
model = AutoModelForTextToWaveform.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, repo_id).to(device)

processor = AutoProcessor.from_pretrained(repo_id)

inputs = processor(
    text=["80s punk and pop track with bassy drums and synth", "punk bossa nova"],
    padding=True,
    return_tensors="pt",
).to(device)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

sampling_rate = model.config.audio_encoder.sampling_rate
audio_values = audio_values.cpu().float().numpy()
sf.write("musicgen_out_0.wav", audio_values[0].T, sampling_rate)
sf.write("musicgen_out_1.wav", audio_values[1].T, sampling_rate)
```

The above snippet simply load the base weights from the original pre-trained checkpoint and then load on-top of it the 100MB of adaptors from the LoRA fine-tuned checkpoint. 

Then, the rest of the code simply use the model it in the same way you'd use MusicGen or MusicGen Melody (refers to the transformers docs [here](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/musicgen#generation) and [here](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/musicgen_melody#generation) respectively). 

Note that this training code also allows fine-tuning without LoRA, in which case you can refer to the transformers docs [here](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/musicgen#generation) for MusicGen and [here](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/musicgen_melody#generation) for MusicGen Melody, in order to do inference on the newly fine-tuned checkpoint.

## FAQ

### Which base checkpoints can I use? Which ones do you recommend?

Here is a quick summary of the MusicGen models that have been trained and released by Meta, and which are compatible with this training code:

| Model                                                                          | Task          | Model size |
|--------------------------------------------------------------------------------|---------------|------------|
| [MusicGen Melody](https://huggingface.co/facebook/musicgen-melody)             | Melody-guided | 1.5B       |
| [MusicGen Melody Large](https://huggingface.co/facebook/musicgen-melody-large) | Melody-guided | 3.3B       |
| [MusicGen Small](https://huggingface.co/facebook/musicgen-small)               | Text-to-music | 300M       |
| [MusicGen Medium](https://huggingface.co/facebook/musicgen-medium)             | Text-to-music | 1.5B       |
| [MusicGen Large](https://huggingface.co/facebook/musicgen-large)               | Text-to-music | 3.3B       |

Additionally, you can track MusicGen models on the hub [here](https://huggingface.co/models?library=transformers&other=musicgen&sort=trending). You'll find some additional checkpoints trained and released by the community, which you can use for inference straight away.

**We recommend using the MusicGen Melody checkpoints, as those are the ones which gave the best results for us.**

### This is difficult to use, do you have simpler ways to do dreambooth?

I'm currently considering adapting the training script to:
1. A hands-on [gradio](https://www.gradio.app/) demo that will require no code. 
2. A notebook, with detailed steps and some explanations, that will require some Python knowledge.

Of course, I welcome all contributions from the community to speed up the implementation of these projects!

### What kind of datasets do I need?

We rely on the [`datasets`](https://huggingface.co/docs/datasets/v2.17.0/en/index) library, which is optimized for speed and efficiency, and is deeply integrated with the [HuggingFace Hub](https://huggingface.co/datasets) which allows easy sharing and loading.

In order to use this repository, you need an audio dataset from [`datasets`](https://huggingface.co/docs/datasets/v2.17.0/en/index) with at least one audio column. You can set `target_audio_column_name` with this column name.

Audio samples must be less than 30 seconds long and contain no lyrics (instrumentals only).

> [!TIP] 
> If you have songs with lyrics, simply set the flag `--audio_separation`, this will performs audio separation thanks to  [`demucs`](https://github.com/adefossez/demucs/tree/main/demucs).
> This is what I've done for the some of my datasets. I've got inspired from [this script](https://github.com/huggingface/dataspeech/blob/main/scripts/filter_audio_separation.py) to do audio separation with `datasets` and `demucs`.

You can also use your own set of descriptions instead of automatically generated ones. In that case, you also need a text column with those descriptions. You can set `text_column_name` with this column name.

### How do I use audio files that I have with your training code?

If you song files in your computer, and want to create a dataset from [`datasets`](https://huggingface.co/docs/datasets/v2.17.0/en/index) with those, you can use easily do this.

1. You first need to create a csv file that contains the full paths to the audio. The column name for those audio files could be for example `audio`.
2. Once you have this csv file, you can load it to a dataset like this:
```python
from datasets import DatasetDict

dataset = DatasetDict.from_csv({"train": PATH_TO_CSV_FILE})
```
3. You then need to convert the audio column name to [`Audio`](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Audio) so that `datasets` understand that it deals with audio files.
```python
from datasets import Audio
dataset = dataset.cast_column("audio", Audio())
```
4. You can then save the datasets [locally](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Dataset.save_to_disk) or [push it to the hub](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub):
```python
dataset.push_to_hub(REPO_ID)
```

Note that you can make the dataset private by passing [`private=True`](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub.private) to the [`push_to_hub`](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub) method. Find other possible arguments [here](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub).


### I'm getting `nan` errors with some checkpoints. What do I do?

There seems to be an error using guidance scale with some musicgen checkpoints. If that happens, I recommend setting `guidance_scale` to 1.0 in the training parameters.


### Can I fine-tune stereo models ?

I haven't tested yet the training script with stereo MusicGen models. I welcome all contributions from the community to test and correct the training script on those!


## License

The code in this repository is released under the Apache license as found in the LICENSE file. The pre-trained MusicGen 
weights are licenced under CC-BY-NC 4.0.

## Acknowledgements

This library builds on top of a number of open-source giants, to whom we'd like to extend our warmest thanks for providing these tools!

Special thanks to:
- The MusicGen team from Meta AI and their [audiocraft](https://github.com/facebookresearch/audiocraft) repository.
- [Nathan Raw](https://github.com/nateraw) for his support and advice.
- the many libraries used, to name but a few: [🤗 datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index), [🤗 accelerate](https://huggingface.co/docs/accelerate/en/index), [wandb](https://wandb.ai/), and [🤗 transformers](https://huggingface.co/docs/transformers/index).
- Hugging Face 🤗 for providing compute resources and time to explore!


## Citation

If you found this repository useful, please consider citing the original MusicGen paper:

```
@misc{copet2024simple,
      title={Simple and Controllable Music Generation}, 
      author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
      year={2024},
      eprint={2306.05284},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
