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
import logging
import time
from typing import Dict

warnings.filterwarnings('ignore')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import toml
from paddle_quantum.finance import CreditRiskAnalyzer


def main(args):
    time_start = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    logging.info(f"Job start at {time_start:s}")

    parsed_configs: Dict = toml.load(args.config)
    
    # input credit portfolio settings
    num_assets = parsed_configs["num_assets"]
    base_default_prob = parsed_configs["base_default_prob"]
    sensitivity = parsed_configs["sensitivity"]
    lgd = parsed_configs["lgd"]
    confidence_level = parsed_configs["confidence_level"]
    degree_of_simulation = parsed_configs["degree_of_simulation"]
    
    estimator = CreditRiskAnalyzer(num_assets, base_default_prob, sensitivity, lgd, 
                                   confidence_level, degree_of_simulation)
    print("The Value at Risk of these assets are", estimator.estimate_var())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Credit risk analysis with paddle quantum.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    main(parser.parse_args())
