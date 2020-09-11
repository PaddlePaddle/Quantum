# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

"""
HGenerator
"""

import scipy
from numpy import trace as np_trace
from paddle_quantum.utils import pauli_str_to_matrix

__all__ = ["H_generator", ]


def H_generator():
    """
    Generate a Hamiltonian with trivial descriptions
    Returns: A Hamiltonian
    """
    # 生成用泡利字符串表示的特定的哈密顿量
    H = [[-1.0, 'z0,z1'], [-1.0, 'z1,z2'], [-1.0, 'z0,z2']]

    # 生成哈密顿量的矩阵信息
    N_SYS_B = 3  # 用于生成吉布斯态的子系统B的量子比特数
    hamiltonian = pauli_str_to_matrix(H, N_SYS_B)

    # 生成理想情况下的目标吉布斯态 rho
    beta = 1.5  # 设置逆温度参数 beta
    rho_G = scipy.linalg.expm(-1 * beta * hamiltonian) / np_trace(scipy.linalg.expm(-1 * beta * hamiltonian))

    # 设置成 Paddle quantum 所支持的数据类型
    hamiltonian = hamiltonian.astype("complex128")
    rho_G = rho_G.astype("complex128")
    return hamiltonian, rho_G