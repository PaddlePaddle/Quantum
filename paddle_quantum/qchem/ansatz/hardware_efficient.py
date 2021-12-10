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
Hardware efficient ansatz 量子电路，以
`Hardware-efficient variational quantum eigensolver
for small molecules and quantum magnets` 方式构建。
具体细节可以参见 https://arxiv.org/abs/1704.05018

该模块依赖于
    - paddlepaddle
    - paddle_quantum
    - ../layers
    - ../qmodel
"""

import paddle
from paddle import nn
from paddle_quantum.circuit import UAnsatz

from ..qmodel import QModel
from .. import layers


class HardwareEfficientModel(QModel):
    r"""面向硬件的量子线路。

    Args:
        num_qubit (int): 量子线路的量子比特数量。
        circuit_depth (int): Cross resonance 和 Euler 转动层的数量。
    """

    def __init__(self, num_qubit: int, circuit_depth: int) -> None:
        super().__init__(num_qubit)

        mid_layers = []
        for i in range(circuit_depth):
            mid_layers.append(layers.CrossResonanceLayer(num_qubit))
            mid_layers.append(layers.EulerRotationLayer(num_qubit))

        self.model = nn.Sequential(
            layers.RotationLayer(num_qubit, "X"),
            layers.RotationLayer(num_qubit, "Z"),
            *mid_layers,
        )

    def forward(
            self,
            state: "paddle.Tensor[paddle.complex128]"
    ) -> "paddle.Tensor[paddle.complex128]":
        r"""运行量子电路

        Args:
            state (paddle.Tensor[paddle.complex128]): 传入量子线路的量子态。

        Returns:
            paddle.Tensor[paddle.complex128]: 运行电路后的量子态
        """

        out = self.model(state)

        cir0 = UAnsatz(self._n_qubit)
        for subcir in self.model:
            cir0 += subcir.circuit
        self._circuit = cir0

        return out
