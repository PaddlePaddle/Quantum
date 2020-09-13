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

from paddle_quantum.utils import random_pauli_str_generator, pauli_str_to_matrix

__all__ = ["H_generator"]


def H_generator(N):
    """
    Generate a Hamiltonian with trivial descriptions
    Returns: A Hamiltonian
    """
    # 生成用泡利字符串表示的随机哈密顿量
    hamiltonian = random_pauli_str_generator(N, terms=10)
    print("Random Hamiltonian in Pauli string format = \n", hamiltonian)

    # 生成哈密顿量的矩阵信息
    H = pauli_str_to_matrix(hamiltonian, N)
    return H
