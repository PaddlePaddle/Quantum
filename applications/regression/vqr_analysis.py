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

r"""
The VQR app cmdline file.
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"

import toml
from paddle_quantum.data_analysis.vqr import QRegressionModel
import paddle_quantum as pq 

pq.set_backend("state_vector")
pq.set_dtype("complex128")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Performing regression analysis on Fish dataset.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    args = parser.parse_args()
    config = toml.load(args.config)
    
    if config["model_name"] == "linear":
        linear_model = QRegressionModel(**config)
        fitted_linear_estimator = linear_model.regression_analyse()

    elif config["model_name"] == "poly":
        poly_model = QRegressionModel(**config)
        fitted_poly_estimator = poly_model.regression_analyse()
    else:
        raise ValueError("Unknown task. Should be either 'linear' or 'poly' for current usage.")
