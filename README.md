# Fine-tune Musicgen

This repository contains a light-weight training code of the transformers implementation of [Musicgen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md),a state-of-the-art controllable text-to-music model. MusicGen is a single stage auto-regressive Transformer model trained over a [32kHz EnCodec tokenizer](https://huggingface.co/facebook/encodec_32khz) with 4 codebooks sampled at 50 Hz.

It aims at giving tools to easily fine-tune the Musicgen suite of models on single small consumer GPUs, and relies on a suite of optimizations and tricks to reduce the resources consumptions.

More precisely, using [LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora#lora), [half-precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format), [gradient accumulation](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps) and [gradient checkpointing](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.gradient_checkpointing), you can fine-tune your own Musicgen Melody [medium](https://huggingface.co/facebook/musicgen-melody) or [large](https://huggingface.co/facebook/musicgen-melody-large) in as little as 15mn on a A100 and as little as 10GB to 16GB of GPU usage.

## 📖 Quick Index
* [Usage](#usage)
* [Training](#training)

## Usage

### Requirements
Simply install transformers, peft and sentence-piece to 


### Inference


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

wandb is optional but better to have it.

### Guide

## TODO
-> make it compatible with both Musicgen and MusicgenMelody

huggingface-cli login

## Thanks

## License