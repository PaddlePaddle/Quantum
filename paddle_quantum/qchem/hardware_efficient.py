# !/usr/bin/env python3
# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
Hardware Efficient ansatz.
"""

from typing import Union, Optional
import paddle
import paddle_quantum as pq

__all__ = ["HardwareEfficientModel"]


class HardwareEfficientModel(pq.gate.Gate):
    r"""
    Args:
        n_qubits: number of qubits.
        depth: depth of the circuit, a layer in Hardware efficient circuit contains [Ry, Rz, CNOT].
        theta: parameters for the Ry and Rz gates inside the circuit.
    """

    def __init__(
            self,
            n_qubits: int,
            depth: int,
            theta: Optional[paddle.Tensor] = None
    ):
        super().__init__(depth, backend=pq.Backend.StateVector)

        layers = []
        if theta is not None:
            assert theta.shape == [n_qubits, 2,
                                   depth], "shape of the parameter should be compatible to n_qubits and depths"
            for d in range(depth - 1):
                layers.append(pq.gate.RY("full", n_qubits, param=theta[:, 0, d]))
                layers.append(pq.gate.RZ("full", n_qubits, param=theta[:, 1, d]))
                layers.append(pq.gate.CNOT("cycle", n_qubits))
            layers.append(pq.gate.RY("full", n_qubits, param=theta[:, 0, depth - 1]))
            layers.append(pq.gate.RZ("full", n_qubits, param=theta[:, 1, depth - 1]))
        else:
            for d in range(depth - 1):
                layers.append(pq.gate.RY("full", n_qubits))
                layers.append(pq.gate.RZ("full", n_qubits))
                layers.append(pq.gate.CNOT("cycle", n_qubits))
            layers.append(pq.gate.RY("full", n_qubits))
            layers.append(pq.gate.RZ("full", n_qubits))

        self.model = pq.ansatz.Sequential(*layers)

    def forward(self, state: pq.State) -> pq.State:
        return self.model(state)
