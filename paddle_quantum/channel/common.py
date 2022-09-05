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
The source file of the classes for several quantum channel.
"""

import paddle
import paddle_quantum
from paddle_quantum.intrinsic import _format_qubits_idx
from .base import Channel
from . import functional
from ..backend import Backend
from typing import Union, Iterable


class BitFlip(Channel):
    r"""A collection of bit flip channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1-p} I,
        E_1 = \sqrt{p} X.

    Args:
        prob: Probability of a bit flip. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, prob: Union[paddle.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__()
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        if isinstance(prob, float):
            self.prob = paddle.to_tensor(prob)
        else:
            self.prob = prob

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubit_idx in self.qubits_idx:
            state = functional.bit_flip(state, self.prob, qubit_idx, self.dtype, self.backend)
        return state


class PhaseFlip(Channel):
    r"""A collection of phase flip channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1 - p} I,
        E_1 = \sqrt{p} Z.

    Args:
        prob: Probability of a phase flip. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, prob: Union[paddle.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__()
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        if isinstance(prob, float):
            self.prob = paddle.to_tensor(prob)
        else:
            self.prob = prob

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubit_idx in self.qubits_idx:
            state = functional.phase_flip(state, self.prob, qubit_idx, self.dtype, self.backend)
        return state


class BitPhaseFlip(Channel):
    r"""A collection of bit phase flip channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1 - p} I,
        E_1 = \sqrt{p} Y.

    Args:
        prob: Probability of a bit phase flip. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, prob: Union[paddle.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__()
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        if isinstance(prob, float):
            self.prob = paddle.to_tensor(prob)
        else:
            self.prob = prob

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubit_idx in self.qubits_idx:
            state = functional.bit_phase_flip(state, self.prob, qubit_idx, self.dtype, self.backend)
        return state


class AmplitudeDamping(Channel):
    r"""A collection of amplitude damping channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 =
        \begin{bmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{bmatrix},
        E_1 =
        \begin{bmatrix}
            0 & \sqrt{\gamma} \\
            0 & 0
        \end{bmatrix}.

    Args:
        gamma: Damping probability. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, gamma: Union[paddle.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__()
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        if isinstance(gamma, float):
            self.gamma = paddle.to_tensor(gamma)
        else:
            self.gamma = gamma

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubit_idx in self.qubits_idx:
            state = functional.amplitude_damping(state, self.gamma, qubit_idx, self.dtype, self.backend)
        return state


class GeneralizedAmplitudeDamping(Channel):
    r"""A collection of generalized amplitude damping channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{p}
        \begin{bmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{bmatrix},
        E_1 = \sqrt{p} \begin{bmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{bmatrix},\\
        E_2 = \sqrt{1-p} \begin{bmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{bmatrix},
        E_3 = \sqrt{1-p} \begin{bmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{bmatrix}.

    Args:
        gamma: Damping probability. Its value should be in the range :math:`[0, 1]`.
        prob: Excitation probability. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, gamma: Union[paddle.Tensor, float], prob: Union[paddle.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__()
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        if isinstance(prob, float):
            self.prob = paddle.to_tensor(prob)
        else:
            self.prob = prob
        if isinstance(gamma, float):
            self.gamma = paddle.to_tensor(gamma)
        else:
            self.gamma = gamma

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubit_idx in self.qubits_idx:
            state = functional.generalized_amplitude_damping(
                state, self.gamma, self.prob, qubit_idx, self.dtype, self.backend)
        return state


class PhaseDamping(Channel):
    r"""A collection of phase damping channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 =
        \begin{bmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{bmatrix},
        E_1 =
        \begin{bmatrix}
            0 & 0 \\
            0 & \sqrt{\gamma}
        \end{bmatrix}.

    Args:
        gamma: Parameter of the phase damping channels. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, gamma: Union[paddle.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__()
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        if isinstance(gamma, float):
            self.gamma = paddle.to_tensor(gamma)
        else:
            self.gamma = gamma

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubit_idx in self.qubits_idx:
            state = functional.phase_damping(state, self.gamma, qubit_idx, self.dtype, self.backend)
        return state


class Depolarizing(Channel):
    r"""A collection of depolarizing channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1-p} I,
        E_1 = \sqrt{p/3} X,
        E_2 = \sqrt{p/3} Y,
        E_3 = \sqrt{p/3} Z.

    Args:
        prob: Parameter of the depolarizing channels. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, prob: Union[paddle.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__()
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        if isinstance(prob, float):
            self.prob = paddle.to_tensor(prob)
        else:
            self.prob = prob

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubit_idx in self.qubits_idx:
            state = functional.depolarizing(state, self.prob, qubit_idx, self.dtype, self.backend)
        return state


class PauliChannel(Channel):
    r"""A collection of Pauli channels.

    Args:
        prob: Probabilities corresponding to the Pauli X, Y, and Z operators. Each value should be in the
            range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.

    Note:
        The sum of three input probabilities should be less than or equal to 1.
    """
    def __init__(
            self, prob: Union[paddle.Tensor, Iterable[float]],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__()
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        if isinstance(prob, Iterable):
            self.prob = paddle.to_tensor(prob)
        else:
            self.prob = prob

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubit_idx in self.qubits_idx:
            state = functional.pauli_channel(state, self.prob, qubit_idx, self.dtype, self.backend)
        return state


class ResetChannel(Channel):
    r"""A collection of reset channels.

    Such a channel reset the state to :math:`|0\rangle` with a probability of p and to :math:`|1\rangle` with
    a probability of q. Its Kraus operators are

    .. math::

        E_0 =
        \begin{bmatrix}
            \sqrt{p} & 0 \\
            0 & 0
        \end{bmatrix},
        E_1 =
        \begin{bmatrix}
            0 & \sqrt{p} \\
            0 & 0
        \end{bmatrix},\\
        E_2 =
        \begin{bmatrix}
            0 & 0 \\
            \sqrt{q} & 0
        \end{bmatrix},
        E_3 =
        \begin{bmatrix}
            0 & 0 \\
            0 & \sqrt{q}
        \end{bmatrix},\\
        E_4 = \sqrt{1-p-q} I.

    Args:
        prob: Probabilities of resetting to :math:`|0\rangle` and to :math:`|1\rangle`. Each value should be
            in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.

    Note:
        The sum of two input probabilities should be less than or equal to 1.
    """
    def __init__(
            self, prob: Union[paddle.Tensor, Iterable[float]],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__()
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        if isinstance(prob, Iterable):
            self.prob = paddle.to_tensor(prob)
        else:
            self.prob = prob

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubit_idx in self.qubits_idx:
            state = functional.reset_channel(state, self.prob, qubit_idx, self.dtype, self.backend)
        return state


class ThermalRelaxation(Channel):
    r"""A collection of thermal relaxation channels.

    Such a channel simulates the mixture of the :math:`T_1` and the :math:`T_2` processes on superconducting devices.

    Args:
        const_t: :math:`T_1` and :math:`T_2` relaxation time in microseconds.
        exec_time: Quantum gate execution time in the process of relaxation in nanoseconds.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.

    Note:
        Relaxation time must satisfy :math:`T_2 \le T_1`. For reference please see https://arxiv.org/abs/2101.02109.
    """
    def __init__(
            self, const_t: Union[paddle.Tensor, Iterable[float]], exec_time: Union[paddle.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__()
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        if isinstance(const_t, float):
            self.const_t = paddle.to_tensor(const_t)
        else:
            self.const_t = const_t
        if isinstance(exec_time, float):
            self.exec_time = paddle.to_tensor(exec_time)
        else:
            self.exec_time = exec_time

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for qubit_idx in self.qubits_idx:
            state = functional.thermal_relaxation(
                state, self.const_t, self.exec_time, qubit_idx, self.dtype, self.backend)
        return state
