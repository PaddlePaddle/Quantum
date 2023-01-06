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
from paddle_quantum.qml.vsql import train, inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify the handwritten digits by the VSQL model.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    args = parser.parse_args()
    config = toml.load(args.config)
    task = config.pop('task')
    if task == 'train':
        train(**config)
    elif task == 'test':
        prediction, prob = inference(**config)
        if config['is_dir']:
            print(f"The prediction results of the input pictures are {str(prediction)[1:-1]} respectively.")
        else:
            prob = prob[0]
            msg = 'For the input image, the model has'
            for idx, item in enumerate(prob):
                if idx == len(prob) - 1:
                    msg += 'and'
                label = config['classes'][idx]
                msg += f' {item:3.2%} confidence that it is {label:d}'
                msg += '.' if idx == len(prob) - 1 else ', '
            print(msg)
    else:
        raise ValueError("Unknown task, it can be train or test.")
