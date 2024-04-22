# Musicgen Dreamboothing

This repository contains lightweight training code for [Musicgen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md), a state-of-the-art controllable text-to-music model. MusicGen is a single stage auto-regressive Transformer model trained over a [32kHz Encodec tokenizer](https://huggingface.co/facebook/encodec_32khz) with 4 codebooks sampled at 50 Hz.

The aim is to provide **tools** to **easily fine-tune** the Musicgen model suite on **small consumer GPUs**, with little data and to leverage a series of optimizations and tricks to reduce resource consumption. The aim is also to easily **share and build** on these fine-tuned models, thanks to [LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora#lora) fine-tuning.

Specifically, this involves:
* using as few data and resources as possible. We're talking fine-tuning with as little as 15mn on an A100 and as little as 10GB to 16GB of GPU utilization.
* easily share and build models thanks to the [HuggingFace hub](https://huggingface.co/models).
* optionally, generate automatic music descriptions
* optionally, training Musicgen is a [Dreambooth](https://huggingface.co/docs/diffusers/en/training/dreambooth)-like style where one key-word 


## 📖 Quick Index
* [Inference](#inference)
* [Training](#training)
* [❓ FAQ](#faq)

## Inference

### Requirements

You actually don't need to install anything from this repository, simply install [transformers](https://huggingface.co/docs/transformers/main/en/index) (from source for now), [PEFT](https://huggingface.co/docs/peft/index) and [sentencepiece](https://github.com/google/sentencepiece) to get ready!

```sh
pip install git+https://github.com/huggingface/transformers peft sentencepiece
```

Additionnally, the following inference snippet also uses soundfile to save the generated music:
```sh
pip install soundfile
```

### Usage

The training code present in this repository offers two options:
- fine-tuning Musicgen without LoRA, in which case you can refer to the transformers docs [here](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/musicgen#generation) for Musicgen and [here](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/musicgen_melody#generation) for Musicgen Melody, in order to do inference on the newly fine-tuned checkpoint.
- fine-tuning Musicgen with LoRA, in which case the following snippet indicates how to generate music:

```python
from peft import PeftConfig, PeftModel
from transformers import AutoModelForTextToWaveform
import torch

repo_id = "ylacombe/musicgen-melody-punk-lora"

config = PeftConfig.from_pretrained(repo_id)
model = AutoModelForTextToWaveform.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, repo_id)
```

You can then use it in the same way you'd use Musicgen or Musicgen Melody (refers to the transformers docs [here](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/musicgen#generation) and [here](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/musicgen_melody#generation) respectively). 

For example, with the previous model, you can generate:

```python
from transformers import AutoProcessor
import soundfile as sf
device = torch.device("cuda:0" if torch.cuda.device_count()>0 else "cpu")
model.to(device)

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

## Training

### Requirements

You first need to clone this repository before installing requirements.

```sh
git clone git@github.com:ylacombe/musicgen-finetuning.git
cd musicgen-finetuning
pip install -r requirements.txt
```

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

### Training guide

The script [`finetune_musicgen.py`](finetune_musicgen.py) is an end-to-end script that:
1. Loads an audio dataset using the [`datasets`](https://huggingface.co/docs/datasets/v2.17.0/en/index) library, for example this [small subset of songs in the punk style](https://huggingface.co/datasets/ylacombe/tiny-punk) derived from the royalty-free [PogChamp Music Classification Competition](https://www.kaggle.com/competitions/kaggle-pog-series-s01e02/overview) dataset.
2. Loads a Musicgen checkpoint from the hub, for example the [1.5B Musicgen Melody checkpoint](https://huggingface.co/facebook/musicgen-melody).
3. (Optional) Generates automatic song descriptions with the `--add_metadata true` flag.  
4. Tokenizes the text descriptions and encode the audio samples.
5. Uses the [transformers' Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) to perform training and evaluation.

You can learn more about the different arguments of the training script by running:

```sh
python finetune_musicgen.py --help
```

To give a practical example, here's how to fine-tune [Musicgen Melody](https://huggingface.co/facebook/musicgen-melody) on 27 minutes of [Punk music](https://huggingface.co/datasets/ylacombe/tiny-punk/viewer/default/clean).

```sh
python finetune_musicgen.py \
    --overwrite_output_dir true \
    --output_dir "./punk_tmp" \
    --dataset_name "ylacombe/tiny-punk" \
    --dataset_config_name "default" \ 
    --target_audio_column_name "others" \ 
    --instance_prompt "punk" \
    --train_split_name "clean" \
    --eval_split_name "clean" \
    --max_duration_in_seconds 30 \
    --min_duration_in_seconds 1.0 \
    --model_name_or_path "facebook/musicgen-melody" \
    --model_revision "refs/pr/14" \
    --preprocessing_num_workers 8 \
    --do_train true \
    --fp16 true \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.1 \
    --dataloader_num_workers 8 \
    --use_lora true \
    --logging_steps 1 \
    --pad_token_id 2048 \
    --decoder_start_token_id 2048 \
    --do_eval true \ 
    --predict_with_generate true \
    --include_inputs_for_metrics true \
    --eval_steps 25 \
    --evaluation_strategy "steps" \
    --per_device_eval_batch_size 1 \
    --max_eval_samples 8 \ 
    --generation_max_length 400 \
    --seed 456 \
    --push_to_hub true \
    --hub_model_id "musicgen-melody-lora-punk" \
```

Using a few tricks, this fine-tuning run used 10GB of GPU memory and ran in under 15 minutes on an A100 GPU.

More specifically, those tricks are [LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora#lora), [half-precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format), [gradient accumulation](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps) and [gradient checkpointing](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.gradient_checkpointing). Note that using the same parameters on [Musicgen Melody large](https://huggingface.co/facebook/musicgen-melody-large) only used 16GB of GPU. 

Also note that you can also use a JSON file to get your parameters. For example, [punk.json](/example_configs/punk.json):

```sh
python finetune_musicgen.py example_configs/punk.json
```

The JSON example above also shows to follow the training thanks to wandb (e.g of what it looks like [here](https://wandb.ai/ylacombe/musicgen_finetuning_experiments/runs/er8zlhzh/workspace?nw=nwuserylacombe)).

### Tips

Some take-aways from the different experiments we've done:
* to fine-tune and keep model ability it's essential to have a low number of epochs.
* for small datasets, a learning rate of 2e-4 gave great results.
* it doesn't actually matter to have the training loss going down, it's always better to actually listen to the output samples.
* you can get quickly get a sense of how and if the model learned by comparing the samples before and after fine-tuning on wandb (e.g [here](https://wandb.ai/ylacombe/musicgen_finetuning_experiments/runs/er8zlhzh/workspace?nw=nwuserylacombe)).
* If you're not using a melody checkpoint and get `nan` errors, you might want to set `guidance_scale` to 1.0, check this [FAQ response](#im-getting-nan-errors-with-some-checkpoints-what-do-i-do).


LoRA is a training technique for significantly reducing the number of trainable parameters. As a result, training is faster and it is easier to store the resulting weights because they are a lot smaller (~100MBs).

## FAQ

### Which base checkpoints can I use? Which ones do you recommend?

Here is a quick summary of the Musicgen models that have been trained and released by Meta, and which are compatible with this training code:

| Model                 | Repository id | Model size |
|-----------------------|---------------|------------|
|       Musicgen Melody |         [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) |   1.5B       |
| Musicgen Melody Large |         [facebook/musicgen-melody-large](https://huggingface.co/facebook/musicgen-melody-large) |  3.3B        |
|        Musicgen Small |         [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) |  300M          |
|       Musicgen Medium |         [facebook/musicgen-medium](https://huggingface.co/facebook/musicgen-medium) |  1.5B          |
| Musicgen Large        |         [facebook/musicgen-large](https://huggingface.co/facebook/musicgen-large) | 3.3B           |

Additionnally, you can track Musicgen models on the hub [here](https://huggingface.co/models?library=transformers&other=musicgent&sort=trending). You'll find some additionnal checkpoints trained and released by the community.

**We recommand using the Musicgen Melody checkpoints, as those are the ones which gave the best results for us.**



### This is difficult to use, do you have simpler ways to do dreambooth?

I'm currently considering adapting the training script to:
1. An hands-on [gradio](https://www.gradio.app/) demo that will require no code. 
2. A notebook, with detailed steps and some explanations, that will require some Python knowledge.

Of course, I welcome all contributions from the community to speed up the implementation of these projects!

### What kind of datasets do I need?

We rely on the [`datasets`](https://huggingface.co/docs/datasets/v2.17.0/en/index) library, which is optimized for speed and efficiency, and is deeply integrated with the [HuggingFace Hub](https://huggingface.co/datasets) which allows easy sharing and loading.

In order to use this repository, you need an audio dataset from [`datasets`](https://huggingface.co/docs/datasets/v2.17.0/en/index) with at least one audio column. You can set `target_audio_column_name` with this column name.

Audio samples must be less than 30 seconds long and contain no lyrics (instrumentals only).

> [!TIPS]
> If you have songs with lyrics, you can use [`demucs`](https://github.com/adefossez/demucs/tree/main/demucs), a model that performs audio separation, to get rid of those.
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


### Why and how do we set `model_revision`?

A discerning eye will have noticed that when using the [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody), we're using the following model revision:

```json
    "model_revision": "refs/pr/14",
```

This is due to a bug in [`transformers`](https://huggingface.co/docs/transformers/index) which has since been corrected, and which made some checkpoints incomplete.

We're thus using the model weights from [this PR](https://huggingface.co/facebook/musicgen-melody/discussions/14), the number 14, to use the correct model weights.

Note that, hopefully soon, you won't need to go through these model revisions once the correct PRs have been merged.

### I'm getting `nan` errors with some checkpoints. What do I do?

There seems to be an error using guidance scale with some musicgen checkpoints. If that happens, I recommend setting `guidance_scale` to 1.0 in the training parameters.


### Can I fine-tune stereo models ?

I haven't tested yet the training script with stereo Musicgen models.

I welcome all contributions from the community to test and correct the training script on those!


## License

The code in this repository is released under the Apache license as found in the LICENSE file.

TODO: how do we specify that the user should look at  model licenses?


## Acknowledgements

This library builds on top of a number of open-source giants, to whom we'd like to extend our warmest thanks for providing these tools!

Special thanks to:
- The Musicgen team from Meta AI and their [audiocraft](https://github.com/facebookresearch/audiocraft) repository.
- [Nathan Raw](https://github.com/nateraw) for his support and advice.
- the many libraries used, to name but a few: [🤗 datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index), [🤗 accelerate](https://huggingface.co/docs/accelerate/en/index), [wandb](https://wandb.ai/), and [🤗 transformers](https://huggingface.co/docs/transformers/index).
- Hugging Face 🤗 for providing compute resources and time to explore!


## Citation

If you found this repository useful, please consider citing the original Musicgen paper:

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
