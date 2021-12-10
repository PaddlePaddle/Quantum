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
Restrict Hartree Fock 模块
"""

from collections import OrderedDict
from copy import deepcopy
import numpy as np
import paddle
from paddle.nn import initializer
from paddle_quantum.circuit import UAnsatz
from ..linalg import givens_decomposition, parameters_to_givens_matrix2
from ..qmodel import QModel
from .. import functional


class _MakeGivensMatrix(paddle.autograd.PyLayer):
    r"""Construct Givens rotation matrix G from parameters \theta and \phi.

    .. math::
        G[j-1,j-1] = \cos(\theta)
        G[j,j] = \cos(\theta)
        G[j-1,j] = -\phi\sin(\theta)
        G[j,j-1] = \phi\sin(\theta)

    We define this function since paddle doesn't fully support differentiate over
    copy and slice operation, e.g. A[i,j] = \theta, where \theta is the parameter 
    to be differentiated.
    """

    @staticmethod
    def forward(ctx, theta: paddle.Tensor, phi: paddle.Tensor, j: int, n: int):
        G = parameters_to_givens_matrix2(theta, phi, j, n)
        ctx.saved_index = j
        ctx.save_for_backward(theta, phi)
        return G

    @staticmethod
    def backward(ctx, dG):
        j = ctx.saved_index
        theta, phi = ctx.saved_tensor()
        dtheta = (-1.0 * (dG[j - 1, j - 1] + dG[j, j]) * paddle.sin(theta)
                  + (dG[j, j - 1] - dG[j - 1, j]) * phi * paddle.cos(theta))
        return dtheta, None


class RestrictHartreeFockModel(QModel):
    r"""限制性 Hartree Fock (RHF) 波函数的量子线路。

    Args:
        num_qubits (int): RHF 计算中需要使用的量子比特数量。
        n_electrons (int): 待模拟的量子化学系统中的电子数量。
        onebody (paddle.Tensor(dtype=paddle.float64)): 经过 L\"owdin 正交化之后的单体积分。
    """

    def __init__(
            self,
            num_qubits: int,
            n_electrons: int,
            onebody: paddle.Tensor
    ) -> None:
        super().__init__(num_qubits)

        self.nocc = n_electrons // 2
        self.norb = num_qubits // 2
        self.nvir = self.norb - self.nocc

        givens_angles = self.get_init_givens_angles(onebody)
        models = []
        for ij, (theta, phi) in givens_angles.items():
            models.append((ij, _GivensBlock(self.n_qubit, -theta, phi)))

        self.models = paddle.nn.Sequential(*models)

    def get_init_givens_angles(self, onebody: paddle.Tensor) -> OrderedDict:
        r"""利用单体积分来初始化 Givens 旋转的角度。

        Args:
            onebody (paddle.Tensor): 经过 L\"owdin 正交化之后的单体积分。

        Returns:
            OrderedDict
        """
        assert type(onebody) == paddle.Tensor, "The onebody integral must be a paddle.Tensor."
        _, U = np.linalg.eigh(onebody.numpy())
        U_tensor = paddle.to_tensor(U)
        return givens_decomposition(U_tensor)

    def forward(self, state: paddle.Tensor) -> paddle.Tensor:
        r"""运行量子电路

        Args:
            state (paddle.Tensor[paddle.complex128]): 传入量子线路的量子态矢量。

        Returns:
            paddle.Tensor[paddle.complex128]: 运行电路后的量子态
        """
        s = deepcopy(state)
        self._circuit = UAnsatz(self.n_qubit)
        for ij, givens_ops in self.models.named_children():
            i, j = [int(p) for p in ij.split(",")]
            s = givens_ops(s, 2 * i, 2 * j)
            self._circuit += givens_ops.circuit
            s = givens_ops(s, 2 * i + 1, 2 * j + 1)
            self._circuit += givens_ops.circuit

        return s

    def single_particle_U(self):
        r"""获取 Hartree Fock 轨道旋转矩阵

        Returns:
            paddle.Tensor, Hartree Fock 轨道旋转矩阵，:math:`n_{orbitals}\times n_{occ}`
        """
        self.register_buffer("_U", paddle.eye(int(self.norb), dtype=paddle.float64))
        for ij, givens_ops in self.models.named_children():
            j = int(ij.split(",")[1])
            self._U = givens_ops.single_particle_U(j, self.norb) @ self._U

        return self._U[:, :self.nocc]
