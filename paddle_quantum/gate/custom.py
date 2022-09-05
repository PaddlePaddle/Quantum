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
The source file of the oracle class and the control oracle class.
"""

import math
import paddle
import paddle_quantum
from . import functional
from .base import Gate
from paddle_quantum.intrinsic import _format_qubits_idx
from typing import Union, Iterable
from paddle_quantum.linalg import is_unitary


class Oracle(Gate):
    """An oracle as a gate.

    Args:
        oracle: Unitary oracle to be implemented.
        qubits_idx: Indices of the qubits on which the gates are applied.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, oracle: paddle.Tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
            num_qubits: int = None, depth: int = 1, gate_name: str = 'O'
    ):
        super().__init__(depth)
        oracle = oracle.cast(paddle_quantum.get_dtype())
        assert is_unitary(oracle), "the input oracle must be a unitary matrix"
        num_acted_qubits = int(math.log2(oracle.shape[0]))
        self.oracle = paddle.cast(oracle, paddle_quantum.get_dtype())
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)
        
        self.gate_name = gate_name

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        for _ in range(0, self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.oracle(state, self.oracle, qubits_idx, self.backend)
        return state


class ControlOracle(Gate):
    """A controlled oracle as a gate.

    Args:
        oracle: Unitary oracle to be implemented.
        qubits_idx: Indices of the qubits on which the gates are applied.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, oracle: paddle.Tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int]],
            num_qubits: int = None, depth: int = 1, gate_name: str = 'cO'
    ) -> None:
        super().__init__(depth)
        complex_dtype = paddle_quantum.get_dtype()
        oracle = oracle.cast(complex_dtype)
        assert is_unitary(oracle), "the input oracle must be a unitary matrix"
        
        num_acted_qubits = int(math.log2(oracle.shape[0]))
        # 暂时只支持单控制位
        oracle = (
            paddle.kron(paddle.to_tensor([[1.0, 0], [0, 0]], dtype=complex_dtype), paddle.eye(2 ** num_acted_qubits)) +
            paddle.kron(paddle.to_tensor([[0.0, 0], [0, 1]], dtype=complex_dtype), oracle)
        )
        num_acted_qubits = num_acted_qubits + 1
        self.oracle = oracle
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)
        
        self.gate_name = gate_name

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        for _ in range(0, self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.oracle(state, self.oracle, qubits_idx, self.backend)
        return state
