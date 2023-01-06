# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
import warnings

warnings.filterwarnings('ignore')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import argparse
import toml
from paddle_quantum.qml.qsann import train, inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify the headlines by the QSANN model.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    args = parser.parse_args()
    config = toml.load(args.config)
    task = config.pop('task')
    if task == 'train':
        train(**config)
    elif task == 'test':
        prediction = inference(**config)
        text = config['text']
        print(f'The input text is {text}.')
        print(f'The prediction of the model is {prediction}.')
    else:
        raise ValueError("Unknown task, it can be train or test.")
