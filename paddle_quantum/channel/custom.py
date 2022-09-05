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
The source file of the classes for custom quantum channels.
"""

import math
import paddle
import paddle_quantum
from .base import Channel
from paddle_quantum.intrinsic import _format_qubits_idx
from . import functional
from typing import Union, Iterable


class KrausRepr(Channel):
    r"""A custom channel in Kraus representation.

    Args:
        kraus_oper: Kraus operators of this channel.
        qubits_idx: Indices of the qubits on which this channel acts.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, kraus_oper: Iterable[paddle.Tensor],
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
            num_qubits: int = None
    ):
        # TODO: need to check whether the input is legal
        super().__init__()
        num_acted_qubits = int(math.log2(kraus_oper[0].shape[0]))
        assert 2 ** num_acted_qubits == kraus_oper[0].shape[0], "The length of oracle should be integer power of 2."
        self.kraus_oper = kraus_oper
        is_single_qubit = True if num_acted_qubits == 1 else False
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)

    def forward(self, state: 'paddle_quantum.State') -> 'paddle_quantum.State':
        for qubits_idx in self.qubits_idx:
            state = functional.kraus_repr(state, self.kraus_oper, qubits_idx, self.dtype, self.backend)
        return state


class ChoiRepr(Channel):
    pass
