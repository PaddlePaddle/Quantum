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
from typing import Dict

warnings.filterwarnings('ignore')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import toml
from paddle_quantum.finance import EuroOptionEstimator


def main(args):
    parsed_configs: Dict = toml.load(args.config)
    
    # input option settings
    initial_price = parsed_configs["initial_price"]
    strike_price = parsed_configs["strike_price"]
    interest_rate = parsed_configs["interest_rate"]
    volatility = parsed_configs["volatility"]
    maturity_date = parsed_configs["maturity_date"]
    degree_of_estimation = parsed_configs["degree_of_estimation"]
    
    estimator = EuroOptionEstimator(initial_price, strike_price, 
                                    interest_rate, volatility, 
                                    maturity_date, degree_of_estimation)
    print("The risk-neutral price of this option is", estimator.estimate())
    print("Below is the circuit realization of this quantum solution.")
    estimator.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="European option pricing with paddle quantum.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    main(parser.parse_args())
