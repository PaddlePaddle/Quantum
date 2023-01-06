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

r"""
Variational Quantum Linear Solver
"""

import argparse
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import toml
import logging
import numpy as np
from paddle_quantum.data_analysis.vqls import compute

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve system of linear equations.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    args = parser.parse_args()
    config = toml.load(args.config)
    A_dir = config.pop('A_dir')
    A = np.load(A_dir)
    b_dir = config.pop('b_dir')
    b = np.load(b_dir)
    result = compute(A, b, **config)

    print('Here is x that solves Ax=b:', result)
    relative_error = np.linalg.norm(b- np.matmul(A,result))/np.linalg.norm(b)
    print('Relative error: ', relative_error)
    logging.basicConfig(
        filename='./linear_solver.log',
        filemode='w',
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO
    )
    msg = f"Relative error: {relative_error}"
    logging.info(msg)
    np.save('./answer.npy', result)