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
The library of representations of channels
"""

import numpy as np

import paddle
from ..base import get_dtype
from ..intrinsic import _zero, _one, _get_float_dtype
from ..linalg import pauli_basis_generation
from ..state import State
from typing import List, Union


def bit_flip_kraus(prob: Union[float, np.ndarray, paddle.Tensor], dtype: str = None) -> List[paddle.Tensor]:
    r"""Kraus representation of a bit flip channel with form

    .. math::

        E_0 = \sqrt{1-p} I,
        E_1 = \sqrt{p} X.

    Args:
        prob: probability :math:`p`.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators

    """
    dtype = get_dtype() if dtype is None else dtype
    prob = prob if isinstance(prob, paddle.Tensor) else paddle.to_tensor(prob, dtype=_get_float_dtype(dtype))
    kraus_oper = [
        [
            paddle.sqrt(1 - prob).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - prob).cast(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob).cast(dtype),
            paddle.sqrt(prob).cast(dtype), _zero(dtype),
        ]
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    return kraus_oper


def phase_flip_kraus(prob: Union[float, np.ndarray, paddle.Tensor], dtype: str = None) -> List[paddle.Tensor]:
    r"""Kraus representation of a phase flip channel with form

    .. math::

        E_0 = \sqrt{1 - p} I,
        E_1 = \sqrt{p} Z.

    Args:
        prob: probability :math:`p`.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators
    """
    dtype = get_dtype() if dtype is None else dtype
    prob = prob if isinstance(prob, paddle.Tensor) else paddle.to_tensor(prob, dtype=_get_float_dtype(dtype))
    kraus_oper = [
        [
            paddle.sqrt(1 - prob).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - prob).cast(dtype),
        ],
        [
            paddle.sqrt(prob).cast(dtype), _zero(dtype),
            _zero(dtype), (-paddle.sqrt(prob)).cast(dtype),
        ]
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    return kraus_oper


def bit_phase_flip_kraus(prob: Union[float, np.ndarray, paddle.Tensor], dtype: str = None) -> List[paddle.Tensor]:
    r"""Kraus representation of a bit-phase flip channel with form

    .. math::

        E_0 = \sqrt{1 - p} I,
        E_1 = \sqrt{p} Y.

    Args:
        prob: probability :math:`p`.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators
    """
    dtype = get_dtype() if dtype is None else dtype
    prob = prob if isinstance(prob, paddle.Tensor) else paddle.to_tensor(prob, dtype=_get_float_dtype(dtype))
    kraus_oper = [
        [
            paddle.sqrt(1 - prob).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - prob).cast(dtype),
        ],
        [
            _zero(dtype), -1j * paddle.sqrt(prob),
            1j * paddle.sqrt(prob), _zero(dtype),
        ]
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    return kraus_oper


def amplitude_damping_kraus(gamma: Union[float, np.ndarray, paddle.Tensor], dtype: str = None) -> List[paddle.Tensor]:
    r"""Kraus representation of an amplitude damping channel with form

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
        gamma: coefficient :math:`\gamma`.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators
    """
    dtype = get_dtype() if dtype is None else dtype
    gamma = gamma if isinstance(gamma, paddle.Tensor) else paddle.to_tensor(gamma, dtype=_get_float_dtype(dtype))
    kraus_oper = [
        [
            _one(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - gamma).cast(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(gamma).cast(dtype),
            _zero(dtype), _zero(dtype)],

    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    return kraus_oper


def generalized_amplitude_damping_kraus(
        gamma: Union[float, np.ndarray, paddle.Tensor],
        prob: Union[float, np.ndarray, paddle.Tensor], dtype: str = None
) -> List[paddle.Tensor]:
    r"""Kraus representation of a generalized amplitude damping channel with form

    .. math::

        E_0 = \sqrt{p} \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{bmatrix},
        E_1 = \sqrt{p} \begin{bmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{bmatrix},\\
        E_2 = \sqrt{1-p} \begin{bmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{bmatrix},
        E_3 = \sqrt{1-p} \begin{bmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{bmatrix}.

    Args:
        gamma: coefficient :math:`\gamma`.
        prob: probability :math:`p`.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators
    """
    dtype = get_dtype() if dtype is None else dtype
    float_dtype = _get_float_dtype(dtype)
    gamma = gamma if isinstance(gamma, paddle.Tensor) else paddle.to_tensor(gamma, dtype=float_dtype)
    prob = prob if isinstance(prob, paddle.Tensor) else paddle.to_tensor(prob, dtype=float_dtype)
    kraus_oper = [
        [
            paddle.sqrt(prob).cast(dtype), _zero(dtype),
            _zero(dtype), (paddle.sqrt(prob).cast(dtype) * paddle.sqrt(1 - gamma)).cast(dtype),
        ],
        [
            _zero(dtype), (paddle.sqrt(prob) * paddle.sqrt(gamma)).cast(dtype),
            _zero(dtype), _zero(dtype),
        ],
        [
            (paddle.sqrt(1 - prob) * paddle.sqrt(1 - gamma)).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - prob).cast(dtype),
        ],
        [
            _zero(dtype), _zero(dtype),
            (paddle.sqrt(1 - prob) * paddle.sqrt(gamma)).cast(dtype), _zero(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    return kraus_oper


def phase_damping_kraus(gamma: Union[float, np.ndarray, paddle.Tensor], dtype: str = None) -> List[paddle.Tensor]:
    r"""Kraus representation of a phase damping channel with form

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
        gamma: coefficient :math:`\gamma`.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators
    """
    dtype = get_dtype() if dtype is None else dtype
    gamma = gamma if isinstance(gamma, paddle.Tensor) else paddle.to_tensor(gamma, dtype=_get_float_dtype(dtype))
    kraus_oper = [
        [
            _one(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - gamma).cast(dtype),
        ],
        [
            _zero(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(gamma).cast(dtype),
        ]
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    return kraus_oper


def depolarizing_kraus(prob: Union[float, np.ndarray, paddle.Tensor], dtype: str = None) -> List[paddle.Tensor]:
    r"""Kraus representation of a depolarizing channel with form

    .. math::

        E_0 = \sqrt{1-3p/4} I,
        E_1 = \sqrt{p/4} X,
        E_2 = \sqrt{p/4} Y,
        E_3 = \sqrt{p/4} Z.

    Args:
        prob: probability :math:`p`.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators
    """
    dtype = get_dtype() if dtype is None else dtype
    prob = prob if isinstance(prob, paddle.Tensor) else paddle.to_tensor(prob, dtype=_get_float_dtype(dtype))
    kraus_oper = [
        [
            paddle.sqrt(1 - 3 * prob / 4).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - 3 * prob / 4).cast(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob / 4).cast(dtype),
            paddle.sqrt(prob / 4).cast(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), -1j * paddle.sqrt(prob / 4).cast(dtype),
            1j * paddle.sqrt(prob / 4).cast(dtype), _zero(dtype),
        ],
        [
            paddle.sqrt(prob / 4).cast(dtype), _zero(dtype),
            _zero(dtype), (-1 * paddle.sqrt(prob / 4)).cast(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    return kraus_oper


def generalized_depolarizing_kraus(prob: float, num_qubits: int, dtype: str = None) -> List[paddle.Tensor]:
    r"""Kraus representation of a generalized depolarizing channel with form

    .. math::

        E_0 = \sqrt{1-(D - 1)p/D} I, \text{ where } D = 4^n,
        E_k = \sqrt{p/D} \sigma_k, \text{ for } 0 < k < D.

    Args:
        prob: probability :math:`p`.
        num_qubits: number of qubits :math:`n` of this channel.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators
    """
    dtype = get_dtype() if dtype is None else dtype
    prob = prob if isinstance(prob, paddle.Tensor) else paddle.to_tensor(prob, dtype=_get_float_dtype(dtype))

    basis = [ele.cast(dtype) * (2 ** num_qubits + 0j) for ele in pauli_basis_generation(num_qubits)]
    I, other_elements = basis[0], basis[1:]

    dim = 4 ** num_qubits
    return (
        [I * (paddle.sqrt(1 - (dim - 1) * prob / dim) + 0j)] +
        [ele * (paddle.sqrt(prob / dim) + 0j) for ele in other_elements]
    )


def pauli_kraus(prob: Union[List[float], np.ndarray, paddle.Tensor], dtype: str = None) -> List[paddle.Tensor]:
    r"""Kraus representation of a pauli channel

    Args:
        prob: a list of three probabilities corresponding to X, Y, Z gate :math:`p`.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators
    """
    dtype = get_dtype() if dtype is None else dtype
    float_dtype = _get_float_dtype(dtype)
    prob = prob.cast(float_dtype) if isinstance(prob, paddle.Tensor) else paddle.to_tensor(prob, dtype=float_dtype)
    prob_x, prob_y, prob_z = prob[0], prob[1], prob[2]
    prob_sum = paddle.sum(prob)
    assert prob_sum <= 1, \
        f"The sum of input probabilities should not be greater than 1: received {prob_sum.item()}"
    prob_i = 1 - prob_sum
    kraus_oper = [
        [
            paddle.sqrt(prob_i).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(prob_i).cast(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob_x).cast(dtype),
            paddle.sqrt(prob_x).cast(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), -1j * paddle.sqrt(prob_y).cast(dtype),
            1j * paddle.sqrt(prob_y).cast(dtype), _zero(dtype),
        ],
        [
            paddle.sqrt(prob_z).cast(dtype), _zero(dtype),
            _zero(dtype), (-paddle.sqrt(prob_z)).cast(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    return kraus_oper


def reset_kraus(prob: Union[List[float], np.ndarray, paddle.Tensor], dtype: str = None) -> List[paddle.Tensor]:
    r"""Kraus representation of a reset channel with form

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
        prob: list of two probabilities of resetting to state :math:`|0\rangle` and :math:`|1\rangle`.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators
    """
    dtype = get_dtype() if dtype is None else dtype
    float_dtype = _get_float_dtype(dtype)
    prob = prob.cast(float_dtype) if isinstance(prob, paddle.Tensor) else paddle.to_tensor(prob, dtype=float_dtype)
    prob_0, prob_1 = prob[0], prob[1]
    prob_sum = paddle.sum(prob)
    assert prob_sum <= 1, \
        f"The sum of input probabilities should not be greater than 1: received {prob_sum.item()}"
    prob_i = 1 - prob_sum
    kraus_oper = [
        [
            paddle.sqrt(prob_0).cast(dtype), _zero(dtype),
            _zero(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob_0).cast(dtype),
            _zero(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), _zero(dtype),
            paddle.sqrt(prob_1).cast(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(prob_1).cast(dtype),
        ],
        [
            paddle.sqrt(prob_i).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(prob_i).cast(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    return kraus_oper


def thermal_relaxation_kraus(
        const_t: Union[List[float], np.ndarray, paddle.Tensor],
        exec_time: Union[List[float], np.ndarray, paddle.Tensor], dtype: str = None
) -> List[paddle.Tensor]:
    r"""Kraus representation of a thermal relaxation channel

    Args:
        const_t: list of :math:`T_1` and :math:`T_2` relaxation time in microseconds.
        exec_time: quantum gate execution time in the process of relaxation in nanoseconds.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a list of Kraus operators.
    """
    dtype = get_dtype() if dtype is None else dtype
    float_dtype = _get_float_dtype(dtype)
    
    const_t = const_t.cast(float_dtype) if isinstance(const_t, paddle.Tensor) else paddle.to_tensor(const_t, dtype=float_dtype)
    t1, t2 = const_t[0], const_t[1]
    assert t2 <= t1, \
        f"The relaxation time T2 and T1 must satisfy T2 <= T1: received T2 {t2} and T1{t1}"
    
    exec_time = exec_time.cast(float_dtype) / 1000 if isinstance(exec_time, paddle.Tensor) else paddle.to_tensor(exec_time / 1000, dtype=float_dtype)
    prob_reset = 1 - paddle.exp(-exec_time / t1)
    prob_z = (1 - prob_reset) * (1 - paddle.exp(-exec_time / t2) * paddle.exp(exec_time / t1)) / 2
    prob_z = _zero(float_dtype) if paddle.abs(prob_z) <= 0 else prob_z
    prob_i = 1 - prob_reset - prob_z
    kraus_oper = [
        [
            paddle.sqrt(prob_i).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(prob_i).cast(dtype),
        ],
        [
            paddle.sqrt(prob_z).cast(dtype), _zero(dtype),
            _zero(dtype), (-paddle.sqrt(prob_z)).cast(dtype),
        ],
        [
            paddle.sqrt(prob_reset).cast(dtype), _zero(dtype),
            _zero(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob_reset).cast(dtype),
            _zero(dtype), _zero(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    return kraus_oper


def replacement_choi(sigma: Union[np.ndarray, paddle.Tensor, State], dtype: str = None) -> paddle.Tensor:
    r"""Choi representation of a replacement channel

    Args:
        sigma: output state of this channel.
        dtype: data type. Defaults to be ``None``.

    Returns:
        a Choi operator.
    """
    dtype = get_dtype() if dtype is None else dtype

    # sanity check
    sigma = sigma if isinstance(sigma, State) else State(sigma)
    sigma.to('density_matrix')
    sigma = sigma.data

    dim = sigma.shape[0]
    return paddle.kron(paddle.eye(dim), sigma).cast(dtype)
