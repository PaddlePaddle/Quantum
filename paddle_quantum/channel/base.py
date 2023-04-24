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
The source file of the basic class for the quantum channels.
"""

import warnings
import math
from typing import Any, Optional, Union, List, Iterable, Callable

import paddle
import paddle_quantum as pq
from . import functional
from ..base import Operator, get_backend, get_dtype
from ..backend import Backend
from ..intrinsic import _get_float_dtype, _format_qubits_idx


class Channel(Operator):
    r"""Basic class for quantum channels.

    Args:
        type_repr: type of a representation. should be ``'Choi'``, ``'Kraus'``, ``'Stinespring'``.
        representation: the representation of this channel. Defaults to ``None`` i.e. not specified.
        qubits_idx: indices of the qubits on which this channel acts on. Defaults to ``None``.
            i.e. list(range(num_acted_qubits)).
        num_qubits: total number of qubits. Defaults to ``None``.
        check_legality: whether check the completeness of the representation if provided. Defaults to ``True``.
        num_acted_qubits: the number of qubits that this channel acts on.  Defaults to ``None``.
        backend: Backend on which the channel is executed. Defaults to ``None``.
        dtype: Type of data. Defaults to ``None``.
        name_scope: Prefix name used by the layer to name parameters. If prefix is "my_layer", parameter name in
            MyLayer can be "my_layer_0.w_n", where "w" is the parameter base name and "n" is an unique suffix auto-generated.
            If ``None``, prefix name will be snake cased class name. Defaults to ``None``.

    Raises:
        ValueError: Unsupported channel representation for ``type_repr``.
        NotImplementedError: The noisy channel can only run in density matrix mode.
        TypeError: Unexpected data type for Channel representation.

    Note:
        If ``representation`` is given, then ``num_acted_qubits`` will be determined by ``representation``, no matter
        ``num_acted_qubits`` is ``None`` or not.
    """
    def __init__(
            self, type_repr: str, representation: Union[paddle.Tensor, List[paddle.Tensor]] = None,
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
            num_qubits: int = None, check_legality: bool = True, num_acted_qubits: int = None,
            backend: Backend = None, dtype: str = None, name_scope: str = None
    ) -> None:
        super().__init__(backend, dtype, name_scope)
        assert representation is not None or num_acted_qubits is not None, (
            "Received None for representation and num_acted_qubits: "
            "either one of them must be specified to initialize a Channel instance.")
        type_repr = type_repr.capitalize()
        if type_repr not in ['Choi', 'Kraus', 'Stinespring']:
            raise ValueError(
                "Unsupported channel representation:"
                f"require 'Choi', 'Kraus' or 'Stinespring', not {type_repr}")

        if representation is None:
            assert num_acted_qubits is not None, (
                "Received None for representation and num_acted_qubits: "
                "either one of them must be specified to initialize a Channel instance.")
        elif type_repr == 'Choi':
            num_acted_qubits = self.__choi_init(representation, check_legality)
        elif type_repr == 'Kraus':
            num_acted_qubits = self.__kraus_init(representation, check_legality)
        elif type_repr == 'Stinespring':
            num_acted_qubits = self.__stinespring_init(representation, check_legality)

        if qubits_idx is None:
            self.qubits_idx = list(range(num_acted_qubits))
        else:
            self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)
        self.type_repr = type_repr
        self.num_acted_qubits = num_acted_qubits

    def __choi_init(self, choi_repr: paddle.Tensor, check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'Choi'``
        """
        if self.backend != Backend.DensityMatrix:
            raise NotImplementedError(
                "The noisy channel can only run in density matrix mode.")

        if not isinstance(choi_repr, paddle.Tensor):
            raise TypeError(
                f"Unexpected data type for Choi representation: expected paddle.Tensor, received {type(choi_repr)}")
        choi_repr = choi_repr.cast(self.dtype)

        num_acted_qubits = int(math.log2(choi_repr.shape[0]) / 2)
        if check_legality:
            # TODO: need to add more sanity check for choi
            assert 2 ** (2 * num_acted_qubits) == choi_repr.shape[0], \
                "The shape of Choi representation should be the integer power of 4: check your inputs."

        self.__choi_repr = choi_repr
        return num_acted_qubits

    def __kraus_init(self, kraus_repr: Union[paddle.Tensor, List[paddle.Tensor]], check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'Kraus'``
        """
        if self.backend != Backend.DensityMatrix:
            raise NotImplementedError(
                "The noisy channel can only run in density matrix mode.")

        complex_dtype = self.dtype
        if isinstance(kraus_repr, paddle.Tensor):
            kraus_repr = [kraus_repr.cast(complex_dtype)]
        elif isinstance(kraus_repr, List):
            kraus_repr = [oper.cast(complex_dtype) for oper in kraus_repr]
        else:
            raise TypeError(
                "Unexpected data type for Kraus representation: "
                f"expected paddle.Tensor or its List, received {type(kraus_repr)}"
            )
        dimension = kraus_repr[0].shape[0]
        num_acted_qubits = int(math.log2(dimension))
        assert 2 ** num_acted_qubits == dimension, \
            "The length of oracle should be integer power of 2."

        # sanity check
        if check_legality:
            oper_sum = paddle.zeros([dimension, dimension]).cast(complex_dtype)
            for oper in kraus_repr:
                oper_sum = oper_sum + paddle.conj(oper.T) @ oper
            err = paddle.norm(paddle.abs(oper_sum - paddle.eye(dimension).cast(complex_dtype))).item()
            if err > min(1e-6 * dimension * len(kraus_repr), 0.01):
                warnings.warn(
                    f"\nThe input data may not be a Kraus representation of a channel: norm(sum(E * E^d) - I) = {err}.",
                    UserWarning)

        self.__kraus_repr = kraus_repr
        return num_acted_qubits

    def __stinespring_init(self, stinespring_repr: paddle.Tensor, check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'Stingspring'``
        """
        if self.backend != Backend.DensityMatrix:
            raise NotImplementedError(
                "The noisy channel can only run in density matrix mode.")

        stinespring_repr = stinespring_repr.cast(self.dtype)
        num_acted_qubits = int(math.log2(stinespring_repr.shape[1]))

        # sanity check
        if check_legality:
            # TODO: need to add more sanity check for stinespring
            dim_ancilla = stinespring_repr.shape[0] // stinespring_repr.shape[1]
            dim_act = stinespring_repr.shape[1]
            assert dim_act * dim_ancilla == stinespring_repr.shape[0], \
                'The width of stinespring matrix should be the factor of its height'

        self.__stinespring_repr = stinespring_repr
        return num_acted_qubits

    def to(
        self, backend: Optional[str] = None, device: Optional[str] = None,
        dtype: Optional[str] = None, blocking: Optional[str] = None
    ) -> None:
        super().to(device, dtype, blocking)
        # TODO: to be implemented
        raise NotImplementedError

    @property
    def choi_repr(self) -> paddle.Tensor:
        r"""Choi representation of a channel

        Returns:
            a tensor with shape :math:`[d_\text{out}^2, d_\text{in}^2]`, where :math:`d_\text{in/out}` is
            the input/output dimension of this channel

        Raises:
            ValueError: Need to specify the Choi representation in this Channel instance.
        """
        type_repr = self.type_repr
        if type_repr == 'Kraus':
            return _kraus_to_choi(self.__kraus_repr)
        elif type_repr == 'Choi':
            if self.__choi_repr is None:
                raise ValueError(
                    "Need to specify the Choi representation in this Channel instance.")
            return self.__choi_repr
        elif type_repr == 'Stinespring':
            return _stinespring_to_choi(self.__stinespring_repr)
        else:
            raise ValueError("Cannot recognize the type_repr, it should be ``'Choi'``, ``'Kraus'``, ``'Stinespring'``.")

    @property
    def kraus_repr(self) -> List[paddle.Tensor]:
        r"""Kraus representation of a channel

        Returns:
            a list of tensors with shape :math:`[d_\text{out}, d_\text{in}]`, where :math:`d_\text{in/out}` is
            the input/output dimension of this channel

        Raises:
            ValueError: Need to specify the Kraus representation in this Channel instance.
        """
        type_repr = self.type_repr
        if type_repr == 'Kraus':
            if self.__kraus_repr is None:
                raise ValueError(
                    "Need to specify the Kraus representation in this Channel instance.")
            return self.__kraus_repr
        elif type_repr == 'Choi':
            return _choi_to_kraus(self.__choi_repr, tol=1e-6)
        elif type_repr == 'Stinespring':
            return _stinespring_to_kraus(self.__stinespring_repr)
        else:
            raise ValueError("Cannot recognize the type_repr, it should be ``'Choi'``, ``'Kraus'``, ``'Stinespring'``.")

    @property
    def stinespring_repr(self) -> paddle.Tensor:
        r"""Stinespring representation of a channel

        Returns:
            a tensor with shape :math:`[r * d_\text{out}, d_\text{in}]`, where :math:`r` is the rank of this channel and
            :math:`d_\text{in/out}` is the input/output dimension of this channel

        Raises:
            ValueError: Need to specify the Stinespring representation in this Channel instance.
        """
        type_repr = self.type_repr
        if type_repr == 'Kraus':
            return _kraus_to_stinespring(self.__kraus_repr)
        elif type_repr == 'Choi':
            return _choi_to_stinespring(self.__choi_repr, tol=1e-6)
        elif type_repr == 'Stinespring':
            if self.__stinespring_repr is None:
                raise ValueError(
                    "Need to specify the Stinespring representation in this Channel instance.")
            return self.__stinespring_repr
        else:
            raise ValueError("Cannot recognize the type_repr, it should be ``'Choi'``, ``'Kraus'``, ``'Stinespring'``.")

    def forward(self, state: pq.State) -> pq.State:
        if state.backend == Backend.QuLeaf:
            raise NotImplementedError
        type_repr = self.type_repr
        for qubits_idx in self.qubits_idx:
            if type_repr in ['Gate', 'Kraus']:
                state = functional.kraus_repr(state, self.kraus_repr, qubits_idx, self.dtype, self.backend)
            elif type_repr == 'Choi':
                state = functional.choi_repr(state, self.choi_repr, qubits_idx, self.dtype, self.backend)
            else:
                state = functional.stinespring_repr(state, self.stinespring_repr, qubits_idx, self.dtype, self.backend)
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
    kraus_oper_tensor = paddle.reshape(
        paddle.concat([paddle.kron(x, x.conj().T) for x in kraus_repr]),
        shape=[len(kraus_repr), ndim, -1]
    )
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
