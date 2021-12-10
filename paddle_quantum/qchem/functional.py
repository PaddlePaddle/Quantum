# !/usr/bin/env python3
# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
用于构建量子电路层的函数操作。
该模块依赖于 paddlepaddle 和 paddle_quantum
"""

import paddle
from paddle_quantum.circuit import UAnsatz


# Rotation layer function
def rot_layer(
        cir: UAnsatz,
        parameters: "paddle.Tensor[paddle.float64]",
        gate_type: str
) -> UAnsatz:
    r"""该函数用来在量子线路上添加一层单比特旋转门，"Rx", "Ry", "Rz"

    Args:
        cir (UAnsatz): 量子线路
        parameters (paddle.Tensor[paddle.float64]): 旋转门的旋转角度
        gate_type (str): "X", "Y" 和 "Z"，例如：如果是 "Rx" 门，则添 "X"

    Returns:
        UAnsatz
    """

    n_qubits = cir.n
    assert n_qubits == len(parameters), \
        "length of the parameters must equal the number of qubits"

    for i in range(n_qubits):
        if gate_type == "X":
            cir.rx(parameters[i], i)
        elif gate_type == "Y":
            cir.ry(parameters[i], i)
        elif gate_type == "Z":
            cir.rz(parameters[i], i)

    return cir


# Euler rotation gate function
def euler_rotation(
        cir: UAnsatz,
        which_qubit: int,
        angles: "paddle.Tensor[paddle.float64]",
) -> UAnsatz:
    r"""该函数定义了单比特的 Euler 旋转门（使用 ZXZ 规范）

    .. math::
        U(\theta,\phi,\gamma)=e^{-i\gamma/2\hat{Z}}e^{-i\phi/2\hat{X}}e^{-i\theta/2\hat{X}}.

    Args:
        cir (UAnsatz): 量子线路
        which (int): Euler 旋转门作用的量子比特编号
        angles (paddle.Tensor[paddle.float64]): Euler 角，存储顺序与 ZXZ 操作的顺序相反。

    Returns:
        UAnsatz
    """
    cir.rz(angles[0], which_qubit)
    cir.rx(angles[1], which_qubit)
    cir.rz(angles[2], which_qubit)
    return cir


# Euler rotation layer function
def euler_rotation_layer(
        cir: UAnsatz,
        parameters: "paddle.Tensor[paddle.float64]",
) -> UAnsatz:
    r"""该函数会在给定的量子线路上添加一层 Euler 旋转门。

    Args:
        cir (UAnsatz): 量子线路。
        parameters (paddle.Tensor[paddle.float64]): Euler 角参数集合。
    """
    n_qubits = cir.n
    assert len(
        parameters) == 3 * n_qubits, "length of parameter should be 3 times of the number of qubits in the circuit."

    for i in range(n_qubits):
        cir = euler_rotation(cir, i, parameters[3 * i:3 * (i + 1)])

    return cir


# Cross resonance gate function
def cross_resonance(
        cir: UAnsatz,
        ctrl_targ: "list[int]",
        phase_angle: paddle.Tensor
) -> UAnsatz:
    r"""该函数定义了一个双比特的 cross resonance (CR) 门。

    .. math::
        U(\theta) = \exp(-i\frac{\theta}{2}\hat{X}\otimes\hat{Z})

    Args:
        cir (UAnsatz): 量子线路。
        ctrl_targ (list[int]): 控制比特和目标比特对应的比特编号。
        phase_angle (paddle.Tensor[paddle.float64]): 旋转角度。

    Returns:
        UAnsatz
    """
    cir.h(ctrl_targ[0])
    cir.rzz(phase_angle, ctrl_targ)
    cir.h(ctrl_targ[0])
    return cir


# Cross resonance layer function
def cr_layer(
        cir: UAnsatz,
        parameters: "paddle.Tensor[paddle.float64]",
        ctrl_qubit_index: "list[int]",
        targ_qubit_index: "list[int]"
) -> UAnsatz:
    """该函数在给定线路上按照给定的控制和目标比特编号添加一层 cross resonance (CR) 门。

    Args:
        cir (UAnsatz): 量子线路。
        parameters (paddle.Tensor[paddle.float64]): CR 门中的角度。
        ctrl_qubit_index (list[int]): 控制比特序号。
        targ_qubit_index (list[int]): 目标比特序号。

    Returns:
        UAnsatz
    """
    assert len(parameters) == len(ctrl_qubit_index) and len(ctrl_qubit_index) == len(targ_qubit_index), \
        "length of parameter must be the same as the number of cr gates"

    for i, ct_index in enumerate(zip(ctrl_qubit_index, targ_qubit_index)):
        cir = cross_resonance(cir, list(ct_index), parameters[i])

    return cir


# Nearest neighbor Givens rotation gate function
def givens_rotation(
        cir: UAnsatz,
        theta: "paddle.Tensor[paddle.float64]",
        q1_index: int,
        q2_index: int
) -> UAnsatz:
    r"""该函数定义了两个相邻比特之间的 Givens 旋转门。详细信息参见 https://arxiv.org/abs/1711.05395.

    Note:
        在 paddlequantum :math:`Ry(\theta)=e^{-i\frac{\theta}{2}\hat{Y}}`.

    Args:
        cir (UAnsatz): 量子线路。
        theta (paddle.Tensor[paddle.float64]): 操作中 Ry 门的角度。
        q1_index (int): 第一个 qubit 的编号。
        q2_index (int): 第二个 qubit 的编号。

    Returns:
        UAnsatz
    """

    cir.cnot([q2_index, q1_index])
    cir.cry(-2 * theta, [q1_index, q2_index])
    cir.cnot([q2_index, q1_index])
    return cir
