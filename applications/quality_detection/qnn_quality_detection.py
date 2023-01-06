# !/usr/bin/env python3
# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

import argparse
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import toml
from paddle_quantum.qml.qnnqd import train, inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect whether there are cracks on the surface of images by the QNNQD model.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    args = parser.parse_args()
    config = toml.load(args.config)
    task = config.pop('task')

    if task == 'train':
        train(**config)
    elif task == 'test':
        prediction, prob, label = inference(**config)
        print(f"The prediction results of the input pictures are {str(prediction)[1:-1]} respectively.")
    else:
        raise ValueError("Unknown task, it can be train or test.")
