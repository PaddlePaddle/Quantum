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
Quantum random number generator.
"""
import os
import warnings
import toml
import argparse
from paddle_quantum.data_analysis.rand_num import random_number_generation


warnings.filterwarnings('ignore')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum random number generation.")
    parser.add_argument(
        "--config", type=str, default='./config.toml', help="The path of toml format config file.")
    
    random_number_generation(**toml.load(parser.parse_args().config))