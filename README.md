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
```

Using a few tricks, this fine-tuning run used 10GB of GPU memory and ran in under 15 minutes on an A100 GPU.

More specifically, those tricks are [LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora#lora), [half-precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format), [gradient accumulation](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps) and [gradient checkpointing](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.gradient_checkpointing). Note that using the same parameters on [Musicgen Melody large](https://huggingface.co/facebook/musicgen-melody-large) only used 16GB of GPU. 

Also note that you can also use a JSON file to get your parameters. For example, [punk.json](/example_configs/punk.json):

```sh
python finetune_musicgen.py example_configs/punk.json
```

### Tips

Some take-aways from the different experiments we've done:
* to fine-tune and keep model ability it's essential to have a low number of epochs
* for small datasets, a learning rate of 2e-4 gave great results
* it doesn't actually matter to have the training loss going down, it's always better to actually listen to the output samples.
* you can get quickly get a sense of how and if the model learned by comparing the samples before and after fine-tuning on wandb (e.g [here](https://wandb.ai/ylacombe/musicgen_finetuning_experiments/runs/er8zlhzh/workspace?nw=nwuserylacombe)).


LoRA is a training technique for significantly reducing the number of trainable parameters. As a result, training is faster and it is easier to store the resulting weights because they are a lot smaller (~100MBs).


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
