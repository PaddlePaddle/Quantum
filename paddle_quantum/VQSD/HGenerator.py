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

from numpy import diag
import scipy

SEED = 13

__all__ = ["generate_rho_sigma", ]


def generate_rho_sigma():
    scipy.random.seed(SEED)
    V = scipy.stats.unitary_group.rvs(4)  # 随机生成一个酉矩阵
    D = diag([0.5, 0.3, 0.1, 0.1])  # 输入目标态 rho 的谱
    V_H = V.conj().T
    rho = V @ D @ V_H  # 生成 rho
    # print(rho)  # 打印量子态 rho

    # 输入用来标记的量子态sigma
    sigma = diag([0.1, 0.2, 0.3, 0.4]).astype('complex128')
    return rho, sigma
