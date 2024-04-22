# Copyright 2024 Yoach Lacombe. All rights reserved.
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

import os
import setuptools


_deps = [
    "transformers",
    "peft",
    "torch",
    "accelerate",
    "sentencepiece",
    "datasets[audio]>=2.12.0",
    "wandb",
    "evaluate",
    "torchaudio",
    "soundfile",
]

_extras_dev_deps = [
    "black~=23.1",
    "isort>=5.5.4",
    "ruff>=0.0.241,<=0.0.259",
]

_extras_metadata_deps = ["msclap", "librosa", "demucs"]

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="musicgen-dreamboothing",
    version=0.1,
    description="Train Musicgen in a DreamBooth style with LoRA.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=_deps,
    extras_require={
        "dev": [_extras_dev_deps],
        "metadata": [_extras_metadata_deps],
    },
)
