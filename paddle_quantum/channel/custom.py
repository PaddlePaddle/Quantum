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
import numpy as np
from typing import Union, Iterable, List

import paddle_quantum
from . import functional
from .base import Channel
from ..backend import Backend
from ..intrinsic import _format_qubits_idx


class ChoiRepr(Channel):
    r"""A custom channel in Choi representation.

    Args:
        choi_repr: Choi operator of this channel.
        qubits_idx: Indices of the qubits on which this channel acts. Defaults to ``None``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
        self,
        choi_repr: paddle.Tensor,
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
        num_qubits: int = None
    ):
        super().__init__('choi', choi_repr, qubits_idx, num_qubits)


class KrausRepr(Channel):
    r"""A custom channel in Kraus representation.

    Args:
        kraus_repr: list of Kraus operators of this channel.
        qubits_idx: Indices of the qubits on which this channel acts. Defaults to ``None``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
        self, kraus_repr: Union[paddle.Tensor, List[paddle.Tensor]], 
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
        num_qubits: int = None
    ):
        super().__init__('kraus', kraus_repr, qubits_idx, num_qubits)


class StinespringRepr(Channel):
    r"""A custom channel in Stinespring representation.

    Args:
        stinespring_mat: Stinespring matrix that represents this channel.
        qubits_idx: Indices of the qubits on which this channel acts. Defaults to ``None``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
        self,
        stinespring_mat: paddle.Tensor,
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
        num_qubits: int = None
    ):
        super().__init__('stinespring', stinespring_mat, qubits_idx, num_qubits)
