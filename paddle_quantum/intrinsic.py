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
The intrinsic function of the paddle quantum.
"""

import numpy as np
import paddle
from typing import Union, Iterable, List


def _zero(dtype):
    return paddle.to_tensor(0, dtype=dtype)


def _one(dtype):
    return paddle.to_tensor(1, dtype=dtype)


def _format_qubits_idx(
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int, str],
        num_qubits: int, num_acted_qubits: int = 1
) -> Union[List[List[int]], List[int]]:
    if num_acted_qubits == 1:
        if qubits_idx == 'full':
            qubits_idx = list(range(0, num_qubits))
        elif qubits_idx == 'even':
            qubits_idx = list(range(0, num_qubits, 2))
        elif qubits_idx == 'odd':
            qubits_idx = list(range(1, num_qubits, 2))
        elif isinstance(qubits_idx, Iterable):
            qubits_idx = list(qubits_idx)
        else:
            qubits_idx = [qubits_idx]
    else:
        if qubits_idx == 'cycle':
            qubits_idx = []
            for idx in range(0, num_qubits - num_acted_qubits):
                qubits_idx.append([i for i in range(idx, idx + num_acted_qubits)])
            for idx in range(num_qubits - num_acted_qubits, num_qubits):
                qubits_idx.append([i for i in range(idx, num_qubits)] + 
                                  [i for i in range(idx + num_acted_qubits - num_qubits)])
        elif qubits_idx == 'linear':
            qubits_idx = []
            for idx in range(0, num_qubits - num_acted_qubits):
                qubits_idx.append([i for i in range(idx, idx + num_acted_qubits)])
        elif len(np.shape(qubits_idx)) == 1 and len(qubits_idx) == num_acted_qubits:
            qubits_idx = [list(qubits_idx)]
        elif len(np.shape(qubits_idx)) == 2 and all((len(indices) == num_acted_qubits for indices in qubits_idx)):
            qubits_idx = [list(indices) for indices in qubits_idx]
        else:
            raise TypeError(
                "The qubits_idx should be iterable such as list, tuple, and so on whose elements are all integers."
                "And the length of acted_qubits should be consistent with the corresponding gate."
            )
    return qubits_idx


def _get_float_dtype(complex_dtype: str) -> str:
    if complex_dtype == 'complex64':
        float_dtype = 'float32'
    elif complex_dtype == 'complex128':
        float_dtype = 'float64'
    else:
        raise ValueError("The dtype should be complex64 or complex128.")
    return float_dtype
