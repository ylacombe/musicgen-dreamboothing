#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Fine-tuning MusicGen for text-to-music using 🤗 Transformers Seq2SeqTrainer"""

import functools
import json
import logging
import os
import re
import sys
import warnings
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForTextToWaveform,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.integrations import is_wandb_available # TODO: keep ?
from accelerate import PartialState



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.38.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")


logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


#### ARGUMENTS

class MusicgenTrainer(Seq2SeqTrainer):
    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length, tensor.shape[2]), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[1]] = tensor
        return padded_tensor


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained processor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    pad_token_id: int = field(
        default=None,
        metadata={"help": "If specified, change the model pad token id."},
    )
    decoder_start_token_id: int = field(
        default=None,
        metadata={"help": "If specified, change the model decoder start token id."},
    )

@dataclass
class DataSeq2SeqTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: str = field(
        default="train+validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to "
                "'train+validation'"
            )
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    target_audio_column_name: str = field( # TODO
        default="audio",
        metadata={"help": "The name of the dataset column containing the target audio data. Defaults to 'audio'"},
    )
    conditional_audio_column_name: str = field( # TODO
        default=None,
        metadata={"help": "The name of the dataset column containing the target audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field( #TODO
        default=None,
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'None'."},
    )
    instance_prompt: str = field( # TODO
        default=None,
        metadata={"help": ""}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    eval_metrics: List[str] = list_field(
        default=[],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    full_generation_sample_text: str = field(
        default="This is a test, let's see what comes out of this.",
        metadata={
            "help": ( # TODO
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    add_audio_samples_to_wandb: bool = field(
        default=False,
        metadata={
            "help": "If set and if `wandb` in args.report_to, will add generated audio samples to wandb logs."
        }
    )
    
@dataclass
class DataCollatorMusicGenWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        # TODO: adapt
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        labels = [torch.tensor(feature["labels"]).transpose(0,1) for feature in features]
        # (bsz, seq_len, num_codebooks)
        labels = torch.nn.utils.rnn.pad_sequence(labels,batch_first=True,padding_value=-100)
        
        
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        input_ids = self.processor.tokenizer.pad(input_ids, return_tensors="pt")
        

        batch= {"labels":labels, **input_ids}
        
        
        if self.feature_extractor_input_name in features[0]:
            # TODO: verify that it works
            input_values = [{self.feature_extractor_input_name: feature[self.feature_extractor_input_name]} for feature in features]
            input_values = self.processor.feature_extractor.pad(input_values, return_tensors="pt")
            
            batch[self.feature_extractor_input_name: input_values]
        
        return batch
    
    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataSeq2SeqTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_musicgen_melody", model_args, data_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. First, let's load the dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
        )

        if data_args.target_audio_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--target_audio_column_name '{data_args.target_audio_column_name}' not found in dataset '{data_args.dataset_name}'."
                " Make sure to set `--target_audio_column_name` to the correct audio column - one of"
                f" {', '.join(raw_datasets['train'].column_names)}."
            )

    
        if data_args.instance_prompt is not None:
            logger.warning(
            f"Using the following instance prompt: {data_args.instance_prompt}"
            )
        elif data_args.text_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(raw_datasets['train'].column_names)}."
            )
        elif data_args.text_column_name is None and data_args.instance_prompt is None:
            raise ValueError(
                f"--instance_prompt or --text_column_name must be set."
            )

        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
        )

        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))


    # 3. Next, let's load the config as we might need it to create
    # load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
    
    # update pad token id and decoder_start_token_id
    config.update({
        "pad_token_id": model_args.pad_token_id if model_args.pad_token_id is not None else model.config.pad_token_id,
        "decoder_start_token_id": model_args.decoder_start_token_id if model_args.decoder_start_token_id is not None else model.config.decoder_start_token_id,
    })
    
    # 4. Now we can instantiate the processor and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load processor
    processor = AutoProcessor.from_pretrained(
        model_args.processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
    
    instance_prompt_tokenized = None
    if data_args.instance_prompt is not None:
        instance_prompt_tokenized = processor.tokenizer(data_args.instance_prompt)

    # create model
    model = AutoModelForTextToWaveform.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
    
    # take audio_encoder_feature_extractor
    audio_encoder_feature_extractor = AutoFeatureExtractor.from_pretrained(
        model.config.audio_encoder._name_or_path,                                                   
        )
    
    # 5. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # TODO: add demucs
    # Drop vocals
    
    # resample target audio
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.target_audio_column_name].sampling_rate
    if dataset_sampling_rate != audio_encoder_feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            data_args.target_audio_column_name, datasets.features.Audio(sampling_rate=audio_encoder_feature_extractor.sampling_rate)
        )

    if data_args.conditional_audio_column_name is not None:
        dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.conditional_audio_column_name].sampling_rate
        if dataset_sampling_rate != processor.feature_extractor.sampling_rate:
            raw_datasets = raw_datasets.cast_column(
                data_args.conditional_audio_column_name, datasets.features.Audio(sampling_rate=processor.feature_extractor.sampling_rate)
            )

    # derive max & min input length for sample rate & max duration
    max_target_length = data_args.max_duration_in_seconds * processor.feature_extractor.sampling_rate
    min_target_length = data_args.min_duration_in_seconds * processor.feature_extractor.sampling_rate
    target_audio_column_name = data_args.target_audio_column_name
    conditional_audio_column_name = data_args.conditional_audio_column_name
    text_column_name = data_args.text_column_name
    num_workers = data_args.preprocessing_num_workers
    feature_extractor_input_name = processor.feature_extractor.model_input_names[0]
    
    if data_args.instance_prompt is not None:
        with training_args.main_process_first(desc="instance_prompt preprocessing"):
            # compute text embeddings on one process since it's only a forward pass
            # do it on CPU for simplicity
            instance_prompt_tokenized = instance_prompt_tokenized["input_ids"]

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_audio_features(batch):
        # load audio
        if conditional_audio_column_name is not None:
            sample = batch[target_audio_column_name]
            inputs = processor.feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
            batch[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
            
        if text_column_name is not None:
            text = batch[text_column_name]
            batch["input_ids"] = processor.tokenizer(text)["input_ids"]
        else:
            batch["input_ids"] = instance_prompt_tokenized

        # load audio
        target_sample = batch[target_audio_column_name]
        labels = audio_encoder_feature_extractor(target_sample["array"], sampling_rate=target_sample["sampling_rate"])
        batch["labels"] = labels["input_values"]
        
        # take length of raw audio waveform
        batch["target_length"] = len(target_sample["array"].squeeze())
        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_audio_features,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_workers,
            desc="preprocess datasets",
        )

        def is_audio_in_length_range(length):
            return length > min_target_length and length < max_target_length

        # filter data that is shorter than min_target_length
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["target_length"],
        )
        
    audio_decoder = model.audio_encoder
    def apply_audio_decoder(batch):
        labels = audio_decoder.encode(torch.tensor(batch["labels"]).to(audio_decoder.device))["audio_codes"]
        labels, delay_pattern_mask = model.decoder.build_delay_pattern_mask(labels, 
                                                model.generation_config.decoder_start_token_id, 
                                                model.generation_config.max_length + 1)
        
        labels = model.decoder.apply_delay_pattern_mask(labels, delay_pattern_mask)
        
        # the first timestamp is associated to a row full of BOS, let's get rid of it
        batch["labels"] = labels[:, 1:]   
        return batch
        
    with training_args.main_process_first(desc="audio target preprocessing"):
        # for now on CPU
        # TODO: enrich for GPU
        vectorized_datasets = vectorized_datasets.map(
            apply_audio_decoder,
            num_proc=num_workers,
            desc="preprocess datasets",
        )
    
    # TODO: will it work on GPU ? unmerged for now https://github.com/huggingface/accelerate/pull/2433
    # for split in vectorized_datasets:
    #     with distributed_state.split_between_processes(vectorized_datasets[split]["labels"]) as input_labels:
    #         result = audio_decoder(input_labels)
    
    if data_args.add_audio_samples_to_wandb and "wandb" in training_args.report_to:
        if is_wandb_available():
            from transformers.integrations import WandbCallback
        else:
            raise ValueError("`args.add_audio_samples_to_wandb=True` but wandb is not installed. See https://docs.wandb.ai/quickstart to install.")
        

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return
    
    # 6. Next, we can prepare the training.
    # Let's use word CLAP similary as our evaluation metric,
    # instantiate a data collator and the trainer

    # Define evaluation metrics during training, *i.e.* CLAP similarity
    # TODO: allow using another CLAP
    clap = AutoModel.from_pretrained("laion/larger_clap_general")
    clap_processor = AutoProcessor.from_pretrained("laion/larger_clap_general")
    
    
    def clap_similarity(texts, audios):
        clap_inputs = clap_processor(text=texts, audios=audios.squeeze(1), return_tensors="pt")
        text_features = clap.get_text_features(clap_inputs["input_ids"], attention_mask=clap_inputs.get("attention_mask", None))
        audio_features = clap.get_audio_features(clap_inputs["input_features"])
        
        cosine_sim = torch.nn.functional.cosine_similarity(audio_features, text_features, dim=1, eps=1e-8)
        
        return cosine_sim.mean()
    
    eval_metrics = {"clap": clap_similarity}

    def compute_metrics(pred):
        input_ids = pred.inputs
        texts = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        audios = pred.predictions
        
        results = {key: metric(texts, audios) for (key, metric) in eval_metrics.items()}

        return results

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            processor.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    data_collator = DataCollatorMusicGenWithPadding(
        processor=processor, feature_extractor_input_name=feature_extractor_input_name
    )
    
    # Freeze Encoders
    model.freeze_encoders()

    # TEST: TODO - add
        
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"], #enc_to_dec_proj, audio_enc_to_dec_proj
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    print(model.targeted_module_names)

        

    # Initialize MusicgenTrainer
    trainer = MusicgenTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=processor,
    )
    
    if data_args.add_audio_samples_to_wandb and "wandb" in training_args.report_to:
        def decode_predictions(processor, predictions):
            audios = predictions.predictions
            return {"audio": np.array(audios.squeeze(1))}


        class WandbPredictionProgressCallback(WandbCallback):
            """Custom WandbCallback to log model predictions during training.
            """
            # TODO: add logging of samples from train set + other things

            def __init__(self, trainer, processor, val_dataset,
                        num_samples=8):
                """Initializes the WandbPredictionProgressCallback instance.

                Args:
                    trainer (Seq2SeqTrainer): The Hugging Face Seq2SeqTrainer instance.
                    processor (AutoProcessor): The processor associated 
                    with the model.
                    val_dataset (Dataset): The validation dataset.
                    num_samples (int, optional): Number of samples to select from 
                    the validation dataset for generating predictions.
                    Defaults to 8.
                """
                super().__init__()
                self.trainer = trainer
                self.processor = processor
                self.sample_dataset = val_dataset.select(range(num_samples))

            def on_train_end(self, args, state, control, **kwargs):
                super().on_train_end(args, state, control, **kwargs)


                predictions = self.trainer.predict(self.sample_dataset)
                # decode predictions and labels
                predictions = decode_predictions(self.processor, predictions)
                
                input_ids = self.sample_dataset["input_ids"]
                texts = self.processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                audios = predictions["audio"]

                # log the table to wandb
                self._wandb.log({"sample_songs": [self._wandb.Audio(audio, caption=text, sample_rate=audio_encoder_feature_extractor.sampling_rate) for (audio, text) in zip(audios, texts)]})



        # Instantiate the WandbPredictionProgressCallback
        progress_callback = WandbPredictionProgressCallback(
            trainer=trainer,
            processor=processor,
            val_dataset=vectorized_datasets["eval"],
            num_samples=2, # TODO: add to args
        )

        # Add the callback to the trainer
        trainer.add_callback(progress_callback)

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:
        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint, ignore_keys_for_eval=["past_key_values", "attentions"])
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    config_name = data_args.dataset_config_name if data_args.dataset_config_name is not None else "na"
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-to-audio",
        "tags": ["text-to-audio", data_args.dataset_name],
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()