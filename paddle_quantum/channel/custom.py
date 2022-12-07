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
import warnings
from typing import Union, Iterable

import paddle_quantum
from .base import Channel
from . import functional
from ..intrinsic import _format_qubits_idx


class KrausRepr(Channel):
    r"""A custom channel in Kraus representation.

    Args:
        kraus_oper: Kraus operators of this channel.
        qubits_idx: Indices of the qubits on which this channel acts.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, kraus_oper: Union[paddle.Tensor, Iterable[paddle.Tensor]],
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
            num_qubits: int = None
    ):
        super().__init__()
        num_acted_qubits = int(math.log2(kraus_oper[0].shape[0]))
        assert 2 ** num_acted_qubits == kraus_oper[0].shape[0], "The length of oracle should be integer power of 2."
        
        self.kraus_oper = [oper.cast(self.dtype) for oper in kraus_oper] if isinstance(kraus_oper, Iterable) else [kraus_oper.cast(self.dtype)]
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)
        
        # sanity check
        dimension = 2 ** num_acted_qubits
        oper_sum = paddle.zeros([dimension, dimension]).cast(self.dtype)
        for oper in self.kraus_oper:
            oper_sum = oper_sum + oper @ paddle.conj(oper.T)
        err = paddle.norm(paddle.abs(oper_sum - paddle.eye(dimension).cast(self.dtype))).item()
        if err > min(1e-6 * dimension * len(kraus_oper), 0.01):
            warnings.warn(
                f"\nThe input data may not be a Kraus representation of a channel: norm(sum(E * E^d) - I) = {err}.", UserWarning)

    def __matmul__(self, other: 'KrausRepr') -> 'KrausRepr':
        r"""Composition between channels with Kraus representations
        
        """
        assert self.qubits_idx == other.qubits_idx, \
            f"Two channels should have the same qubit indices to composite: received {self.qubits_idx} and {other.qubits_idx}"
        if not isinstance(other, KrausRepr):
            raise NotImplementedError(
                f"does not support the composition between KrausRepr and {type(other)}")
        new_kraus_oper = []
        for this_kraus in self.kraus_oper:
            new_kraus_oper.extend([this_kraus @ other_kraus for other_kraus in other.kraus_oper])
        return KrausRepr(new_kraus_oper, self.qubits_idx)
    
    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        for qubits_idx in self.qubits_idx:
            state = functional.kraus_repr(state, self.kraus_oper, qubits_idx, self.dtype, self.backend)
        return state


class ChoiRepr(Channel):
    def __init__(
        self,
        choi_oper: paddle.Tensor,
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
        num_qubits: int = None
    ):
        super().__init__()
        num_acted_qubits = int(math.log2(choi_oper.shape[0]) / 2)
        assert 2 ** (2 * num_acted_qubits) == choi_oper.shape[0], "The length of oracle should be integer power of 2."
        self.choi_oper = choi_oper
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        for qubits_idx in self.qubits_idx:
            state = functional.choi_repr(
                state, 
                self.choi_oper, 
                qubits_idx,
                self.dtype,
                self.backend
            )
        return state


class StinespringRepr(Channel):
    def __init__(
        self,
        stinespring_matrix: paddle.Tensor,
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
        num_qubits: int = None
    ):
        super().__init__()
        num_acted_qubits = int(math.log2(stinespring_matrix.shape[1]))
        dim_ancilla = stinespring_matrix.shape[0] // stinespring_matrix.shape[1]
        dim_act = stinespring_matrix.shape[1]
        assert dim_act * dim_ancilla == stinespring_matrix.shape[0], 'The width of stinespring matrix should be the factor of its height'
        self.stinespring_matrix = stinespring_matrix
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)
    
    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        for qubits_idx in self.qubits_idx:
            state = functional.stinespring_repr(
                state, 
                self.stinespring_matrix, 
                qubits_idx,
                self.dtype,
                self.backend
            )
        return state
