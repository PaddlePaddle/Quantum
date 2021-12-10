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
量子电路层，该模块依赖于 paddlepaddle，paddle_quantum 和 math 包。
同时依赖于 qmodel 和 functional 模块。
"""

import math
import paddle
from paddle.nn import initializer

from paddle_quantum.circuit import UAnsatz

from .qmodel import QModel
from . import functional


# Rotation layer
class RotationLayer(QModel):
    r"""单比特旋转门层。

    Args:
        num_qubits (int): 量子线路的量子比特数。
        gate_type (str): 量子门的类型，"X", "Y" 和 "Z" 中的一个。
        trainable (bool): 层中的角度参数是否可训练。
    """
    def __init__(self, num_qubits: int, gate_type: str, trainable: bool = True):
        super().__init__(num_qubits)

        self._angle_attr = paddle.ParamAttr(
            initializer=initializer.Uniform(0.0, 2*math.pi),
            trainable=trainable
        )

        self.angles = self.create_parameter(
            shape=[num_qubits],
            attr=self._angle_attr,
            dtype="float64"
        )

        self._gate_type = gate_type

    def forward(
            self,
            state: "paddle.Tensor[paddle.complex128]"
    ) -> "paddle.Tensor[paddle.complex128]":
        r"""获取运行后的量子态

        Args:
            state (paddle.Tensor[paddle.complex128, shape=[n]]): 送入电路的量子态

        Returns:
            paddle.Tensor[paddle.complex128, shape=[n]]: 运行电路后的量子态
        """
        cir0 = UAnsatz(self._n_qubit)
        self._circuit = functional.rot_layer(cir0, self.angles, self._gate_type)
        return self.circuit.run_state_vector(state)

    def extra_repr(self):
        r"""额外表示
        """
        return "gate={:s}, dtype={:s}".format(
            self._gate_type,
            self.angles.dtype.name)


# Euler rotation layer
class EulerRotationLayer(QModel):
    r"""Euler 旋转门层。

    Args:
        num_qubits (int): 量子线路中的量子比特数量。
        trainable (bool): 层中的参数是否是可训练的。
    """

    def __init__(self, num_qubits: int, trainable: bool = True) -> None:
        super().__init__(num_qubits)
        self._angle_attr = paddle.ParamAttr(
            initializer=initializer.Uniform(0.0, 2*math.pi),
            trainable=trainable
        )
        self.euler_angles = self.create_parameter(
            shape=[3*num_qubits],
            attr=self._angle_attr,
            dtype="float64"
        )

    def forward(
            self,
            state: "paddle.Tensor[paddle.complex128]"
    ) -> "paddle.Tensor[paddle.complex128]":
        r"""获取运行后的量子态

        Args:
            state (paddle.Tensor[paddle.complex128, shape=[n]]): 送入电路的量子态

        Returns:
            paddle.Tensor[paddle.complex128, shape=[n]]: 运行电路后的量子态
        """
        cir0 = UAnsatz(self._n_qubit)
        self._circuit = functional.euler_rotation_layer(cir0, self.euler_angles)
        return self._circuit.run_state_vector(state)

    def extra_repr(self):
        r"""额外表示
        """
        return "dtype={:s}".format(self.euler_angles.dtype.name)


# Cross resonance layer
class CrossResonanceLayer(QModel):
    r"""在量子线路中按照给定的控制和目标比特添加一层 cross resonance 门。

    Args:
        num_qubits (int): 量子比特数目。
        ctrl_qubit_index (list[int]): 控制比特的序号。
        targ_qubit_index (list[int]): 目标比特的序号。
        trainable (bool): 层中的参数是否可训练。
    """
    def __init__(
            self,
            num_qubits: int,
            ctrl_qubit_index: "list[int]" = None,
            targ_qubit_index: "list[int]" = None,
            trainable: bool = True
    ) -> None:
        super().__init__(num_qubits)

        if ctrl_qubit_index is None:
            ctrl_qubit_index = list(range(num_qubits))
        if targ_qubit_index is None:
            targ_qubit_index = list(range(1, num_qubits)) + [0]

        self._ctrl_qubit_index = ctrl_qubit_index
        self._targ_qubit_index = targ_qubit_index

        self._phase_attr = paddle.ParamAttr(
            initializer=initializer.Uniform(0.0, 2*math.pi),
            trainable=trainable
        )
        self.phase = self.create_parameter(
            shape=[len(ctrl_qubit_index)],
            attr=self._phase_attr,
            dtype="float64"
        )

    def forward(
            self,
            state: "paddle.Tensor[paddle.complex128]"
    ) -> "paddle.Tensor[paddle.complex128]":
        r"""获取运行后的量子态

        Args:
            state (paddle.Tensor[paddle.complex128, shape=[n]]): 送入电路的量子态

        Returns:
            paddle.Tensor[paddle.complex128, shape=[n]]: 运行电路后的量子态
        """
        cir0 = UAnsatz(self._n_qubit)
        self._circuit = functional.cr_layer(
            cir0,
            self.phase,
            self._ctrl_qubit_index,
            self._targ_qubit_index)
        return self._circuit.run_state_vector(state)

    def extra_repr(self):
        r"""额外表示
        """
        return "dtype={:s}".format(self.phase.dtype.name)
