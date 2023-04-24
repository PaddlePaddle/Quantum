# !/usr/bin/env python3
# Copyright (c) 2023 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

r"""
Handwritten digit generation via quantum-circuit associative adversarial networks (QCAAN)
"""

import os
import warnings

import argparse
import toml
from paddle_quantum.qml.qcaan import train, model_test

warnings.filterwarnings('ignore')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generating the handwritten digits by the QC-AAN model.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    args = parser.parse_args()
    config = toml.load(args.config)
    mode = config.pop('mode')
    if mode == 'train':
        train(**config)
    elif mode == 'inference':
        model_test(**config)
    else:
        raise ValueError("Unknown mode, it can be 'train' or 'inference'.")
