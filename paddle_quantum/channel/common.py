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
The source file of the classes for common quantum channels.
"""

import paddle
from .custom import KrausRepr
from ..intrinsic import _format_qubits_idx
from ..qinfo import kraus_unitary_random
from .representation import *
from typing import Union, Iterable


class BitFlip(KrausRepr):
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
        super().__init__(bit_flip_kraus(prob), qubits_idx, num_qubits, check_complete=False)
        

class PhaseFlip(KrausRepr):
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
        super().__init__(phase_flip_kraus(prob), qubits_idx, num_qubits, check_complete=False)


class BitPhaseFlip(KrausRepr):
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
        super().__init__(bit_phase_flip_kraus(prob), qubits_idx, num_qubits, check_complete=False)


class AmplitudeDamping(KrausRepr):
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
        super().__init__(amplitude_damping_kraus(gamma), qubits_idx, num_qubits, check_complete=False)


class GeneralizedAmplitudeDamping(KrausRepr):
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
        super().__init__(generalized_amplitude_damping_kraus(gamma, prob), qubits_idx, num_qubits, check_complete=False)


class PhaseDamping(KrausRepr):
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
        super().__init__(phase_damping_kraus(gamma), qubits_idx, num_qubits, check_complete=False)


class Depolarizing(KrausRepr):
    r"""A collection of depolarizing channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1-3p/4} I,
        E_1 = \sqrt{p/4} X,
        E_2 = \sqrt{p/4} Y,
        E_3 = \sqrt{p/4} Z.

    Args:
        prob: Parameter of the depolarizing channels. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.

    Note:
        The implementation logic for this feature has been updated.
        The current version refers to formula (8.102) in Quantum Computation and Quantum Information 10th
        edition by M.A.Nielsen and I.L.Chuang.
        Reference: Nielsen, M., & Chuang, I. (2010). Quantum Computation and Quantum Information: 10th
        Anniversary Edition. Cambridge: Cambridge University Press. doi:10.1017/CBO9780511976667
    """
    def __init__(
            self, prob: Union[paddle.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        super().__init__(depolarizing_kraus(prob), qubits_idx, num_qubits, check_complete=False)


class GeneralizedDepolarizing(KrausRepr):
    r"""A generalized depolarizing channel.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1-(D - 1)p/D} I, \text{ where } D = 4^n, \\
        E_k = \sqrt{p/D} \sigma_k, \text{ for } 0 < k < D.
    
    Args:
        prob: probability :math:`p`. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act, the length of which is :math:`n`.
        num_qubits: Total number of qubits. Defaults to ``None``.
    
    """
    def __init__(
            self, prob: Union[paddle.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str], num_qubits: int = None
    ):
        num_acted_qubits = np.size(np.array(qubits_idx)).item()
        super().__init__(generalized_depolarizing_kraus(prob, num_acted_qubits),
                         qubits_idx, num_qubits, check_complete=False)


class PauliChannel(KrausRepr):
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
        super().__init__(pauli_kraus(prob), qubits_idx, num_qubits, check_complete=False)


class ResetChannel(KrausRepr):
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
        super().__init__(reset_kraus(prob), qubits_idx, num_qubits, check_complete=False)


class ThermalRelaxation(KrausRepr):
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
        super().__init__(thermal_relaxation_kraus(const_t, exec_time), qubits_idx, num_qubits, check_complete=False)


class MixedUnitaryChannel(KrausRepr):
    r"""A collection of single-qubit mixed unitary channels.

    Such a channel's Kraus operators are randomly generated unitaries times related probabilities
    .. math::

        N(\rho) = \sum_{i}  p_{i} U_{i} \rho U_{i}^{\dagger}

    Args:
        num_unitary: The amount of random unitaries to be generated.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.

    Note:
    The probability distribution of unitaries is set to be uniform distribution.
    """
    def __init__(
            self, num_unitary: int,
            qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ):
        #TODO: increase the number of acting qubits, currently only support 1
        super().__init__(kraus_unitary_random(1, num_unitary), 
                         _format_qubits_idx(qubits_idx, num_qubits, 1), 
                         num_qubits, check_complete=False)
