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
        qubits_idx: Indices of the qubits on which this channel acts.
        num_qubits: Total number of qubits. Defaults to ``None``.
        
    Raises:
        NotImplementedError: The noisy channel can only run in density matrix mode.
    
    """
    def __init__(
        self,
        choi_repr: paddle.Tensor,
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
        num_qubits: int = None
    ):
        super().__init__()
        num_acted_qubits = int(math.log2(choi_repr.shape[0]) / 2)
        assert 2 ** (2 * num_acted_qubits) == choi_repr.shape[0], "The length of oracle should be integer power of 2."
        
        #TODO: need to add sanity check for choi
        self.__choi_repr = choi_repr
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)
        
        if self.backend != Backend.DensityMatrix:
            raise NotImplementedError(
                "The noisy channel can only run in density matrix mode.")
        
    @property
    def choi_repr(self) -> paddle.Tensor:
        r"""Choi representation
        """
        return self.__choi_repr
    
    @property
    def kraus_repr(self) -> List[paddle.Tensor]:
        r"""Kraus representation
        """
        return _choi_to_kraus(self.__choi_repr, tol=1e-6)
    
    @property
    def stinespring_repr(self) -> paddle.Tensor:
        r"""Stinespring representation
        """
        return _choi_to_stinespring(self.__choi_repr, tol=1e-6)
    
    def to_kraus(self) -> 'KrausRepr':
        r"""Convert to Kraus representation of this chanel        
        """
        return KrausRepr(self.kraus_repr, qubits_idx=self.qubits_idx)
    
    def to_stinespring(self) -> 'StinespringRepr':
        r"""Convert to Stinespring representation of this chanel        
        """
        return StinespringRepr(self.stinespring_repr, qubits_idx=self.qubits_idx)

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubits_idx in self.qubits_idx:
            state = functional.choi_repr(state, self.__choi_repr, qubits_idx, self.dtype, self.backend)
        return state


class KrausRepr(Channel):
    r"""A custom channel in Kraus representation.

    Args:
        kraus_repr: list of Kraus operators of this channel.
        qubits_idx: Indices of the qubits on which this channel acts.
        num_qubits: Total number of qubits. Defaults to ``None``.
        check_complete: whether check the kraus representation is valid. Defaults to be ``True``. 
        Set to ``False`` only if the data correctness is guaranteed.
    
    Raises:
        NotImplementedError: The noisy channel can only run in density matrix mode.
    
    """
    def __init__(
        self, kraus_repr: Union[paddle.Tensor, List[paddle.Tensor]], qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
        num_qubits: int = None, check_complete: bool = True
    ):
        super().__init__()
        num_acted_qubits = int(math.log2(kraus_repr[0].shape[0]))
        assert 2 ** num_acted_qubits == kraus_repr[0].shape[0], "The length of oracle should be integer power of 2."
        
        # kraus operation formalize
        self.__kraus_repr = [oper.cast(self.dtype) for oper in kraus_repr] if isinstance(kraus_repr, List) else [kraus_repr.cast(self.dtype)]
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)
        
        # sanity check
        if check_complete:
            dimension = 2 ** num_acted_qubits
            oper_sum = paddle.zeros([dimension, dimension]).cast(self.dtype)
            for oper in self.__kraus_repr:
                oper_sum = oper_sum + oper @ paddle.conj(oper.T)
            err = paddle.norm(paddle.abs(oper_sum - paddle.eye(dimension).cast(self.dtype))).item()
            if err > min(1e-6 * dimension * len(kraus_repr), 0.01):
                warnings.warn(
                    f"\nThe input data may not be a Kraus representation of a channel: norm(sum(E * E^d) - I) = {err}.", UserWarning)
            
        if self.backend != Backend.DensityMatrix:
            raise NotImplementedError(
                "The noisy channel can only run in density matrix mode.")

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

    @property
    def choi_repr(self) -> paddle.Tensor:
        r"""Choi representation
        """
        return _kraus_to_choi(self.__kraus_repr)
    
    @property
    def kraus_repr(self) -> List[paddle.Tensor]:
        r"""Kraus representation
        """
        return self.__kraus_repr
    
    @property
    def stinespring_repr(self) -> paddle.Tensor:
        r"""Stinespring representation
        """
        return _kraus_to_stinespring(self.__kraus_repr)
    
    def to_choi(self) -> 'ChoiRepr':
        r"""Convert to Choi representation of this chanel        
        """
        return ChoiRepr(self.choi_repr, qubits_idx=self.qubits_idx)
    
    def to_stinespring(self) -> 'StinespringRepr':
        r"""Convert to Stinespring representation of this chanel        
        """
        return StinespringRepr(self.stinespring_repr, qubits_idx=self.qubits_idx)
    
    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubits_idx in self.qubits_idx:
            state = functional.kraus_repr(state, self.__kraus_repr, qubits_idx, self.dtype, self.backend)
        return state


class StinespringRepr(Channel):
    r"""A custom channel in Stinespring representation.

    Args:
        stinespring_mat: Stinespring matrix that represents this channel.
        qubits_idx: Indices of the qubits on which this channel acts.
        num_qubits: Total number of qubits. Defaults to ``None``.
    
    Raises:
        NotImplementedError: The noisy channel can only run in density matrix mode.
    
    """
    def __init__(
        self,
        stinespring_mat: paddle.Tensor,
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
        num_qubits: int = None
    ):
        super().__init__()
        num_acted_qubits = int(math.log2(stinespring_mat.shape[1]))
        dim_ancilla = stinespring_mat.shape[0] // stinespring_mat.shape[1]
        dim_act = stinespring_mat.shape[1]
        assert dim_act * dim_ancilla == stinespring_mat.shape[0], 'The width of stinespring matrix should be the factor of its height'
        
        #TODO: need to add sanity check for stinespring
        self.__stinespring_repr = stinespring_mat
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)
        
        if self.backend != Backend.DensityMatrix:
            raise NotImplementedError(
                "The noisy channel can only run in density matrix mode.")
        
    @property
    def choi_repr(self) -> paddle.Tensor:
        r"""Choi representation
        """
        return _stinespring_to_choi(self.__stinespring_repr)
    
    @property
    def kraus_repr(self) -> List[paddle.Tensor]:
        r"""Kraus representation
        """
        return _stinespring_to_kraus(self.__stinespring_repr)
    
    @property
    def stinespring_repr(self) -> paddle.Tensor:
        r"""Stinespring representation
        """
        return self.__stinespring_repr
    
    def to_choi(self) -> 'ChoiRepr':
        r"""Convert to Choi representation of this chanel        
        """
        return ChoiRepr(self.choi_repr, qubits_idx=self.qubits_idx)
    
    def to_kraus(self) -> 'KrausRepr':
        r"""Convert to Kraus representation of this chanel        
        """
        return KrausRepr(self.kraus_repr, qubits_idx=self.qubits_idx)

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubits_idx in self.qubits_idx:
            state = functional.stinespring_repr(state, self.__stinespring_repr, qubits_idx, self.dtype, self.backend)
        return state
    

def _choi_to_kraus(choi_repr: paddle.Tensor, tol: float) -> List[paddle.Tensor]:
    r"""Transform the Choi representation to the Kraus representation
    """
    ndim = int(math.sqrt(choi_repr.shape[0]))
    w, v = paddle.linalg.eigh(choi_repr)

    # add abs to make eigvals safe
    w = paddle.abs(w)
    l_cut = 0
    for l in range(len(w) - 1, -1, -1):
        if paddle.sum(paddle.abs(w[l:])) / paddle.sum(paddle.abs(w)) > 1 - tol:
            l_cut = l
            break
    return [(v * paddle.sqrt(w))[:, l].reshape([ndim, ndim]).T for l in range(l_cut, ndim**2)]


def _choi_to_stinespring(choi_repr: paddle.Tensor, tol: float) -> List[paddle.Tensor]:
    r"""Transform the Choi representation to the Stinespring representation
    """
    # TODO: need a more straightforward transformation
    return _kraus_to_stinespring(_choi_to_kraus(choi_repr, tol))


def _kraus_to_choi(kraus_repr: List[paddle.Tensor]) -> paddle.Tensor:
    r"""Transform the Kraus representation to the Choi representation
    """
    ndim = kraus_repr[0].shape[0]
    kraus_oper_tensor = paddle.concat([paddle.kron(x, x.conj().T) for x in kraus_repr]).reshape([len(kraus_repr), ndim, -1])
    choi_repr = paddle.sum(kraus_oper_tensor, axis=0).reshape([ndim for _ in range(4)]).transpose([2, 1, 0, 3])
    return choi_repr.transpose([0, 2, 1, 3]).reshape([ndim * ndim, ndim * ndim])


def _kraus_to_stinespring(kraus_repr: List[paddle.Tensor]) -> paddle.Tensor:
    r"""Transform the Kraus representation to the Stinespring representation
    """
    j_dim = len(kraus_repr)
    i_dim = kraus_repr[0].shape[0]
    kraus_oper_tensor = paddle.concat(kraus_repr).reshape([j_dim, i_dim, -1])
    stinespring_repr = kraus_oper_tensor.transpose([1, 0, 2])
    return stinespring_repr.reshape([i_dim * j_dim, i_dim])


def _stinespring_to_choi(stinespring_repr: paddle.Tensor) -> paddle.Tensor:
    r"""Transform the Stinespring representation to the Choi representation
    """
    # TODO: need a more straightforward transformation
    return _kraus_to_choi(_stinespring_to_kraus(stinespring_repr))


def _stinespring_to_kraus(stinespring_repr: paddle.Tensor) -> List[paddle.Tensor]:
    r"""Transform the Stinespring representation to the Kraus representation
    """
    i_dim = stinespring_repr.shape[1]
    j_dim = stinespring_repr.shape[0] // i_dim
    kraus_oper = stinespring_repr.reshape([i_dim, j_dim, i_dim]).transpose([1, 0, 2])
    return [kraus_oper[j] for j in range(j_dim)]
